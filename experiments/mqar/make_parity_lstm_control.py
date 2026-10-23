from pathlib import Path

source = Path("train_parity_baseline.py")
output = Path("train_parity_lstm_control.py")
text = source.read_text(encoding="utf-8")

start = text.index("class CausalSelfAttention")
end = text.index("\n\nclass TransformerBlock", start)

replacement = '''class CausalSelfAttention(SequenceMixer):
    """Nonlinear recurrent positive control for parity."""

    def __init__(self):
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
    '"parity_standard_transformer_baseline"',
    '"parity_lstm_control"',
)
text = text.replace(
    "results/parity_baseline",
    "results/parity_lstm_control",
)
text = text.replace(
    "PARITY BASELINE:",
    "PARITY LSTM CONTROL:",
)

output.write_text(text, encoding="utf-8")
print("Created:", output.resolve())
