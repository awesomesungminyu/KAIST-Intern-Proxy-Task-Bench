from pathlib import Path

source = Path("train_flipflop_transformer.py")
output = Path("train_flipflop_lstm.py")
text = source.read_text(encoding="utf-8")

start = text.index("class CausalSelfAttention")
end = text.index("\n\nclass TransformerBlock", start)

replacement = '''class CausalSelfAttention(SequenceMixer):
    """Nonlinear recurrent positive control."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=CONFIG["d_model"],
            hidden_size=CONFIG["d_model"],
            num_layers=1,
            batch_first=True,
        )
        self.out_proj = nn.Linear(
            CONFIG["d_model"],
            CONFIG["d_model"],
            bias=False,
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.out_proj(output)
'''

text = text[:start] + replacement + text[end:]
text = text.replace(
    "flipflop_standard_transformer",
    "flipflop_lstm_control",
)
text = text.replace(
    "results/flipflop_transformer",
    "results/flipflop_lstm",
)
text = text.replace(
    "FLIP-FLOP TRANSFORMER:",
    "FLIP-FLOP LSTM:",
)

output.write_text(text, encoding="utf-8")
print("Created:", output.resolve())
