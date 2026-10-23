import os
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


CONFIG = {
    "run": "wikitext2_transformer_plus_baseline",
    "dataset": "Salesforce/wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "tokenizer": "gpt2",
    "context_length": 256,
    "batch_size": 32,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 384,
    "dropout": 0.1,
    "learning_rate": 0.0005,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_epochs": 15,
    "early_stop_patience": 3,
    "seed": 456,
    "dtype": "float32",
}

# _HEXA_T_SMOKE_OVERRIDES_V1
_HEXA_T_INTEGER_OVERRIDES = {
    "HEXA_T_MAX_EPOCHS": "max_epochs",
    "HEXA_T_BATCH_SIZE": "batch_size",
}
for _hexa_env, _hexa_key in _HEXA_T_INTEGER_OVERRIDES.items():
    if _hexa_env in os.environ:
        CONFIG[_hexa_key] = int(os.environ[_hexa_env])

# _HEXA_T_INTEGRATION_V2
if "HEXA_T_SEED" in os.environ:
    CONFIG["seed"] = int(os.environ["HEXA_T_SEED"])

ROOT = Path("/data/mqar")
CACHE_DIR = ROOT / "data/wikitext2_gpt2"
OUTPUT_DIR = ROOT / "results/wikitext2_transformer_baseline"
if os.environ.get("HEXA_T_OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["HEXA_T_OUTPUT_DIR"])
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(scale + self.eps) * self.weight


def apply_rope(q, k):
    length = q.size(-2)
    head_dim = q.size(-1)

    positions = torch.arange(
        length, device=q.device, dtype=torch.float32
    )
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=q.device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )

    frequencies = torch.outer(positions, inv_freq)
    cosine = frequencies.cos()[None, None, :, :]
    sine = frequencies.sin()[None, None, :, :]

    def rotate(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack(
            (
                even * cosine - odd * sine,
                even * sine + odd * cosine,
            ),
            dim=-1,
        ).flatten(-2)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch,
                length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        q, k = apply_rope(q, k)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout.p if self.training else 0.0
            ),
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, length, dim)
        return self.dropout(self.output(output))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.gate(x)) * self.value(x)
        return self.dropout(self.output(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = RMSNorm(CONFIG["d_model"])
        self.sequence_mixer = CausalSelfAttention(
            CONFIG["d_model"],
            CONFIG["num_heads"],
            CONFIG["dropout"],
        )
        self.ffn_norm = RMSNorm(CONFIG["d_model"])
        self.ffn = SwiGLU(
            CONFIG["d_model"],
            CONFIG["ffn_dim"],
            CONFIG["dropout"],
        )

    def forward(self, x):
        x = x + self.sequence_mixer(
            self.attention_norm(x)
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """Adapter for an official MAD architecture recipe."""

    def __init__(self, vocab_size=None):
        super().__init__()

        import copy as _copy
        import json as _json
        import os as _os
        import sys as _sys
        from pathlib import Path as _Path

        import yaml as _yaml

        _root = _Path("/data/mqar")
        _mad = _root / "vendor" / "mad-lab"

        if str(_mad) not in _sys.path:
            _sys.path.insert(0, str(_mad))

        from mad.registry import layer_registry as _registry
        from mad.model.language_model import LanguageModel as _LanguageModel

        _model_id = _os.environ["HEXA_T_MAD_MODEL_ID"]
        _recipes = _json.loads(
            (
                _root
                / "configs"
                / "hexa_t"
                / "mad16_recipes.json"
            ).read_text(encoding="utf-8")
        )["models"]

        _recipe = next(
            item for item in _recipes
            if item["id"] == _model_id
        )

        _vocab_size = int(
            vocab_size
            if vocab_size is not None
            else CONFIG["vocab_size"]
        )
        _max_length = int(
            CONFIG.get(
                "context_length",
                CONFIG.get("sequence_length", 64),
            )
        )
        _dimension = int(CONFIG["d_model"])

        _modules = []
        _configs = []

        for _layer_name in _recipe["layers"]:
            _entry = _registry[_layer_name]
            _config = _yaml.safe_load(
                (_mad / _entry["cfg"]).read_text(
                    encoding="utf-8"
                )
            )
            _config = _copy.deepcopy(_config)

            if "dim" in _config:
                _config["dim"] = _dimension
            if "d_model" in _config:
                _config["d_model"] = _dimension

            _config["max_length"] = _max_length
            if "l_max" in _config:
                _config["l_max"] = _max_length

            _modules.append(_entry["module"])
            _configs.append(_config)

        self.inner = _LanguageModel(
            vocab_size=_vocab_size,
            layers=_modules,
            layer_cfgs=_configs,
            dim=_dimension,
            max_length=_max_length,
            position_embeds=__import__("torch").nn.Embedding,
        )
        self.mad_model_id = _model_id
        self.mad_layers = list(_recipe["layers"])

    def forward(self, token_ids):
        output = self.inner(token_ids)
        if isinstance(output, tuple):
            output = output[0]
        return output


class TokenBlocks(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length
        self.num_blocks = (
            (len(tokens) - 1) // context_length
        )

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        start = index * self.context_length
        end = start + self.context_length

        return (
            self.tokens[start:end],
            self.tokens[start + 1 : end + 1],
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_tokens(dataset, tokenizer, split):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{split}.pt"

    if cache_path.exists():
        print("Loading token cache:", cache_path)
        return torch.load(cache_path)

    print("Tokenizing:", split)
    raw_text = "".join(dataset[split]["text"])

    token_ids = tokenizer(
        raw_text,
        add_special_tokens=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    token_ids.append(tokenizer.eos_token_id)

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
    )
    torch.save(tokens, cache_path)
    return tokens


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for inputs, labels in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(inputs)
        token_count = labels.numel()

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        total_loss += loss.item() * token_count
        total_tokens += token_count

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity


set_seed(CONFIG["seed"])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading dataset...")
dataset = load_dataset(
    CONFIG["dataset"],
    CONFIG["dataset_config"],
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["tokenizer"],
    use_fast=True,
    clean_up_tokenization_spaces=False,
)
tokenizer.model_max_length = 10**30

train_tokens = prepare_tokens(
    dataset,
    tokenizer,
    "train",
)
valid_tokens = prepare_tokens(
    dataset,
    tokenizer,
    "validation",
)

train_dataset = TokenBlocks(
    train_tokens,
    CONFIG["context_length"],
)
valid_dataset = TokenBlocks(
    valid_tokens,
    CONFIG["context_length"],
)

generator = torch.Generator()
generator.manual_seed(CONFIG["seed"])

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    generator=generator,
    num_workers=0,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

model = TransformerLM(len(tokenizer)).to(device)

decay_parameters = []
no_decay_parameters = []

for name, parameter in model.named_parameters():
    if parameter.ndim >= 2 and "embedding" not in name:
        decay_parameters.append(parameter)
    else:
        no_decay_parameters.append(parameter)

optimizer = torch.optim.AdamW(
    [
        {
            "params": decay_parameters,
            "weight_decay": CONFIG["weight_decay"],
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ],
    lr=CONFIG["learning_rate"],
)

total_steps = len(train_loader) * CONFIG["max_epochs"]


def lr_multiplier(step):
    if step < CONFIG["warmup_steps"]:
        return (step + 1) / CONFIG["warmup_steps"]

    progress = (
        step - CONFIG["warmup_steps"]
    ) / max(
        1,
        total_steps - CONFIG["warmup_steps"],
    )

    return 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_multiplier,
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

print("device:", device)
print("train tokens:", len(train_tokens))
print("valid tokens:", len(valid_tokens))
print("train blocks:", len(train_dataset))
print("valid blocks:", len(valid_dataset))
print("train batches:", len(train_loader))
print("parameters:", total_parameters)

history = []
best_valid_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
global_steps = 0
total_start = time.perf_counter()

for epoch in range(1, CONFIG["max_epochs"] + 1):
    model.train()

    epoch_start = time.perf_counter()
    train_loss_sum = 0.0
    train_token_count = 0

    for batch_index, (inputs, labels) in enumerate(
        train_loader,
        start=1,
    ):
        inputs = inputs.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        token_count = labels.numel()
        train_loss_sum += loss.item() * token_count
        train_token_count += token_count
        global_steps += 1

        if batch_index % 50 == 0:
            print(
                f"epoch={epoch:02d}",
                f"batch={batch_index:03d}/{len(train_loader)}",
                f"loss={loss.item():.4f}",
                flush=True,
            )

    train_loss = train_loss_sum / train_token_count
    valid_loss, valid_ppl = evaluate(
        model,
        valid_loader,
        device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    epoch_seconds = time.perf_counter() - epoch_start
    tokens_per_second = (
        train_token_count / epoch_seconds
    )

    improved = valid_loss < best_valid_loss - 1e-4

    if improved:
        best_valid_loss = valid_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": CONFIG,
                "vocab_size": len(tokenizer),
                "best_valid_loss": best_valid_loss,
                "best_valid_ppl": valid_ppl,
                "epoch": epoch,
            },
            CHECKPOINT_PATH,
        )
    else:
        epochs_without_improvement += 1

    history.append(
        {
            "epoch": epoch,
            "global_steps": global_steps,
            "train_loss": train_loss,
            "train_ppl": math.exp(
                min(train_loss, 20.0)
            ),
            "valid_loss": valid_loss,
            "valid_ppl": valid_ppl,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": epoch_seconds,
            "tokens_per_second": tokens_per_second,
        }
    )

    print(
        f"epoch={epoch:02d}",
        f"train_loss={train_loss:.4f}",
        f"valid_loss={valid_loss:.4f}",
        f"valid_ppl={valid_ppl:.2f}",
        f"tokens/s={tokens_per_second:,.0f}",
        f"seconds={epoch_seconds:.1f}",
        f"best_epoch={best_epoch}",
        flush=True,
    )

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    if (
        epochs_without_improvement
        >= CONFIG["early_stop_patience"]
    ):
        print("Early stopping.")
        break

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed_seconds = time.perf_counter() - total_start
peak_vram_mib = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 2)
    if device.type == "cuda"
    else 0.0
)

best_valid_ppl = math.exp(
    min(best_valid_loss, 20.0)
)

summary = {
    "run": CONFIG["run"],
    "passed": math.isfinite(best_valid_loss),
    "config": CONFIG,
    "best_epoch": best_epoch,
    "best_valid_loss": best_valid_loss,
    "best_valid_ppl": best_valid_ppl,
    "epochs_completed": len(history),
    "global_steps": global_steps,
    "elapsed_seconds": elapsed_seconds,
    "tokens_per_second_last_epoch": history[-1][
        "tokens_per_second"
    ],
    "device": str(device),
    "gpu": (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else None
    ),
    "compute_capability": (
        list(torch.cuda.get_device_capability(0))
        if device.type == "cuda"
        else None
    ),
    "torch_version": torch.__version__,
    "dtype": "torch.float32",
    "total_parameters": total_parameters,
    "peak_vram_mib": peak_vram_mib,
    "checkpoint": str(CHECKPOINT_PATH),
    "history": str(HISTORY_PATH),
    "test_evaluated": False,
}
summary.update({
    "architecture": "MAD architecture",
    "mad_model_id": __import__("os").environ.get(
        "HEXA_T_MAD_MODEL_ID"
    ),
    "mad_layers": getattr(model, "mad_layers", None),
    "mad_commit":
        "0f49a452b84ca0d13f8eb9c1ffa649032376fb1b",
    "attention_backend":
        "torch-sdpa-turing-compatibility",
    "hexa_t_task": "H4",
})


SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n================================")
print("WIKITEXT-2 BASELINE: COMPLETE")
print("best epoch:", best_epoch)
print("best validation loss:", best_valid_loss)
print("best validation PPL:", best_valid_ppl)
print("elapsed minutes:", elapsed_seconds / 60)
print("peak VRAM MiB:", peak_vram_mib)
print("checkpoint:", CHECKPOINT_PATH)
print("summary:", SUMMARY_PATH)
