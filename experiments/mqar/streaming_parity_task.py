import json
from pathlib import Path

import torch


BIT_ZERO = 1
BIT_ONE = 2
QUERY = 3
ANSWER_START = 4
IGNORE = -100


def make_dataset(num_examples, sequence_length, modulus, seed):
    generator = torch.Generator().manual_seed(seed)

    bits = torch.randint(
        low=0,
        high=2,
        size=(num_examples, sequence_length - 1),
        generator=generator,
    )

    bit_tokens = torch.where(bits == 0, BIT_ZERO, BIT_ONE)
    query_tokens = torch.full(
        (num_examples, 1),
        QUERY,
        dtype=torch.long,
    )
    inputs = torch.cat([bit_tokens, query_tokens], dim=1)

    prefix_answers = (
        bits.cumsum(dim=1).remainder(modulus) + ANSWER_START
    )
    final_answer = prefix_answers[:, -1:]
    labels = torch.cat([prefix_answers, final_answer], dim=1)

    return inputs.long(), labels.long()


def main():
    records = []

    for length in [32, 64, 128, 256]:
        inputs, labels = make_dataset(
            num_examples=2000,
            sequence_length=length,
            modulus=2,
            seed=987654321,
        )

        bits = (inputs[:, :-1] == BIT_ONE).long()
        expected_prefix = (
            bits.cumsum(dim=1).remainder(2) + ANSWER_START
        )

        assert inputs.shape == labels.shape
        assert torch.equal(labels[:, :-1], expected_prefix)
        assert torch.equal(labels[:, -1], expected_prefix[:, -1])
        assert torch.all(inputs[:, -1] == QUERY)
        assert not torch.any(inputs >= ANSWER_START)

        oracle_accuracy = torch.cat(
            [expected_prefix, expected_prefix[:, -1:]],
            dim=1,
        ).eq(labels).float().mean().item()

        records.append({
            "length": length,
            "examples": 2000,
            "oracle_accuracy": oracle_accuracy,
        })

        print(
            f"length={length:3d}",
            f"oracle_accuracy={oracle_accuracy:.4f}",
        )

    result_path = Path(
        "/data/mqar/results/streaming_parity_data_validation.json"
    )
    result_path.write_text(
        json.dumps(
            {
                "run": "streaming_parity_data_validation",
                "passed": True,
                "supervision": "prefix parity at every position",
                "results": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("==============================")
    print("STREAMING PARITY DATA VALIDATION: PASS")
    print("saved:", result_path)


if __name__ == "__main__":
    main()
