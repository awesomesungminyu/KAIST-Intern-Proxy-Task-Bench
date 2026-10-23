import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from zoology.data.multiquery_ar import MQARConfig


CONFIG = {
    "vocab_size": 8192,
    "sequence_length": 64,
    "num_kv_pairs": 8,
    "train_examples": 20_000,
    "valid_examples": 2_000,
    "batch_size": 64,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 512,
    "learning_rate": 1e-3,
    "weight_decay": 0.1,
    "max_epochs": 30,
    "early_stop_accuracy": 0.99,
    "data_seed_train": 123,
    "data_seed_valid": 2_147_483_789,
    "model_seed": int(os.environ.get("HARNESS_SEED", "456")),
    "dtype": "float32",
}

_HARNESS_INT_OVERRIDES = {
    "HARNESS_SEQUENCE_LENGTH": "sequence_length",
    "HARNESS_NUM_KV_PAIRS": "num_kv_pairs",
    "HARNESS_TRAIN_EXAMPLES": "train_examples",
    "HARNESS_VALID_EXAMPLES": "valid_examples",
    "HARNESS_BATCH_SIZE": "batch_size",
    "HARNESS_MAX_EPOCHS": "max_epochs",
}

for _environment_name, _config_name in _HARNESS_INT_OVERRIDES.items():
    if _environment_name in os.environ:
        CONFIG[_config_name] = int(os.environ[_environment_name])



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceMixer(nn.Module):
    def forward(self, x):
        raise NotImplementedError


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10_000):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k):
        length = q.size(2)
        positions = torch.arange(
            length, device=q.device, dtype=torch.float32
        )
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos().to(q.dtype)[None, None]
        sin = angles.sin().to(q.dtype)[None, None]

        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack(
                (
                    even * cos - odd * sin,
                    even * sin + odd * cos,
                ),
                dim=-1,
            ).flatten(-2)

        return rotate(q), rotate(k)


class CausalSelfAttention(SequenceMixer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        q, k = self.rope(q, k)

        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        output = output.transpose(1, 2).contiguous().view(
            batch, length, width
        )
        return self.output(output)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer_norm = nn.LayerNorm(CONFIG["d_model"])
        self.sequence_mixer = CausalSelfAttention(
            CONFIG["d_model"], CONFIG["num_heads"]
        )
        self.ffn_norm = nn.LayerNorm(CONFIG["d_model"])
        self.ffn = nn.Sequential(
            nn.Linear(CONFIG["d_model"], CONFIG["ffn_dim"]),
            nn.GELU(),
            nn.Linear(CONFIG["ffn_dim"], CONFIG["d_model"]),
        )

    def forward(self, x):
        x = x + self.sequence_mixer(self.mixer_norm(x))
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



def make_dataset(num_examples, seed):
    config = MQARConfig(
        vocab_size=CONFIG["vocab_size"],
        num_examples=num_examples,
        input_seq_len=CONFIG["sequence_length"],
        num_kv_pairs=CONFIG["num_kv_pairs"],
        random_non_queries=True,
        include_slices=True,
    )
    data = config.build(seed=seed)
    return TensorDataset(data.inputs, data.labels)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    target_count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        mask = labels != -100
        targets = labels[mask]
        selected_logits = logits[mask]

        loss_sum += (
            F.cross_entropy(
                selected_logits, targets, reduction="sum"
            ).item()
        )
        correct += (
            selected_logits.argmax(dim=-1) == targets
        ).sum().item()
        target_count += targets.numel()

    return loss_sum / target_count, correct / target_count


seed_everything(CONFIG["model_seed"])
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Generating training dataset...")
train_dataset = make_dataset(
    CONFIG["train_examples"], CONFIG["data_seed_train"]
)

print("Generating validation dataset...")
valid_dataset = make_dataset(
    CONFIG["valid_examples"], CONFIG["data_seed_valid"]
)

loader_generator = torch.Generator()
loader_generator.manual_seed(CONFIG["model_seed"])

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    generator=loader_generator,
    num_workers=0,
    pin_memory=device.type == "cuda",
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
)

model = TransformerLM().to(device=device, dtype=torch.float32)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)

parameter_groups = {
    "all_trainable": sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
}
total_parameters = sum(
    p.numel() for p in model.parameters()
)

output_dir = Path(os.environ.get("HARNESS_OUTPUT_DIR", "results/mqar_baseline"))
output_dir.mkdir(parents=True, exist_ok=True)
history_path = output_dir / "history.csv"
checkpoint_path = output_dir / "best_model.pt"
summary_path = output_dir / "summary.json"

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

print("device:", device)
print("parameters:", total_parameters)
print("train batches:", len(train_loader))
print("valid batches:", len(valid_loader))

start_time = time.perf_counter()
best_accuracy = -1.0
best_epoch = 0
best_loss = None
global_step = 0
history = []

for epoch in range(1, CONFIG["max_epochs"] + 1):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_targets = 0

    epoch_start = time.perf_counter()

    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)

        mask = labels != -100
        targets = labels[mask]
        selected_logits = logits[mask]

        loss = F.cross_entropy(selected_logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        count = targets.numel()
        train_loss_sum += loss.item() * count
        train_correct += (
            selected_logits.argmax(dim=-1) == targets
        ).sum().item()
        train_targets += count
        global_step += 1

    valid_loss, valid_accuracy = evaluate(
        model, valid_loader, device
    )

    train_loss = train_loss_sum / train_targets
    train_accuracy = train_correct / train_targets
    epoch_seconds = time.perf_counter() - epoch_start

    row = {
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
        "epoch_seconds": epoch_seconds,
    }
    history.append(row)

    print(
        f"epoch={epoch:02d} "
        f"train_loss={train_loss:.5f} "
        f"train_acc={train_accuracy:.4f} "
        f"valid_loss={valid_loss:.5f} "
        f"valid_acc={valid_accuracy:.4f} "
        f"seconds={epoch_seconds:.1f}"
    )

    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_loss = valid_loss
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": CONFIG,
                "epoch": epoch,
                "global_step": global_step,
                "valid_accuracy": valid_accuracy,
                "valid_loss": valid_loss,
            },
            checkpoint_path,
        )

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=history[0].keys()
        )
        writer.writeheader()
        writer.writerows(history)

    if valid_accuracy >= CONFIG["early_stop_accuracy"]:
        print("Early-stop target reached.")
        break

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed_seconds = time.perf_counter() - start_time
peak_vram_bytes = (
    torch.cuda.max_memory_allocated()
    if device.type == "cuda"
    else 0
)

try:
    zoology_commit = subprocess.check_output(
        [
            "git", "-C", "/data/mqar/zoology-src",
            "rev-parse", "HEAD",
        ],
        text=True,
    ).strip()
except Exception:
    zoology_commit = None

summary = {
    "run": "mqar_standard_transformer_baseline",
    "passed": best_accuracy >= CONFIG["early_stop_accuracy"],
    "config": CONFIG,
    "best_epoch": best_epoch,
    "best_valid_loss": best_loss,
    "best_valid_accuracy": best_accuracy,
    "epochs_completed": epoch,
    "global_steps": global_step,
    "elapsed_seconds": elapsed_seconds,
    "steps_per_second": global_step / elapsed_seconds,
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
    "torch_cuda_version": torch.version.cuda,
    "dtype": "torch.float32",
    "total_parameters": total_parameters,
    "parameter_groups": parameter_groups,
    "weight_tying": True,
    "position_encoding": "RoPE inside attention",
    "sequence_mixer_interface": "(B, T, D) -> (B, T, D)",
    "peak_vram_bytes": peak_vram_bytes,
    "peak_vram_mib": peak_vram_bytes / (1024 ** 2),
    "zoology_commit": zoology_commit,
    "checkpoint": str(checkpoint_path.resolve()),
    "history": str(history_path.resolve()),
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
    "hexa_t_task": "H1",
})

summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n==============================")
print(
    "BASELINE RUN:",
    "PASS" if summary["passed"] else "TARGET NOT REACHED",
)
print("best epoch:", best_epoch)
print("best valid accuracy:", best_accuracy)
print("total parameters:", total_parameters)
print("elapsed minutes:", round(elapsed_seconds / 60, 2))
print(
    "peak VRAM MiB:",
    round(peak_vram_bytes / (1024 ** 2), 2),
)
print("checkpoint:", checkpoint_path.resolve())
print("history:", history_path.resolve())
print("summary:", summary_path.resolve())
