import json
from pathlib import Path

import torch


RESULT = Path("/data/mqar/results/parity_data_validation.json")

BIT_ZERO = 1
BIT_ONE = 2
QUERY = 3
ANSWER_START = 4
IGNORE = -100


def make_dataset(num_examples, sequence_length, modulus, seed):
    generator = torch.Generator().manual_seed(seed)

    bits = torch.randint(
        0,
        2,
        (num_examples, sequence_length - 1),
        generator=generator,
    )

    inputs = torch.where(bits == 0, BIT_ZERO, BIT_ONE)
    inputs = torch.cat(
        [inputs, torch.full((num_examples, 1), QUERY)],
        dim=1,
    )

    answers = bits.sum(dim=1).remainder(modulus) + ANSWER_START

    labels = torch.full_like(inputs, IGNORE)
    labels[:, -1] = answers

    return inputs.long(), labels.long()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    tests = []

    for length in (32, 64, 128):
        for modulus in (2, 4, 8):
            x, y = make_dataset(1000, length, modulus, seed=123)
            same_x, same_y = make_dataset(1000, length, modulus, seed=123)
            other_x, _ = make_dataset(1000, length, modulus, seed=456)

            expected = (
                (x[:, :-1] == BIT_ONE).sum(dim=1).remainder(modulus)
                + ANSWER_START
            )

            check(x.shape == (1000, length), "input shape error")
            check(y.shape == x.shape, "label shape error")
            check(torch.equal(x, same_x), "same seed is not deterministic")
            check(torch.equal(y, same_y), "same seed labels differ")
            check(not torch.equal(x, other_x), "different seeds match")
            check(torch.all(x[:, -1] == QUERY), "query position error")
            check(torch.all(y[:, :-1] == IGNORE), "masking error")
            check(torch.equal(y[:, -1], expected), "answer error")
            check(torch.all(x < ANSWER_START), "answer leaked into input")

            tests.append({
                "sequence_length": length,
                "modulus": modulus,
                "examples": 1000,
                "supervised_targets": int((y != IGNORE).sum()),
                "passed": True,
            })

    sample_x, sample_y = make_dataset(3, 16, 2, seed=7)

    result = {
        "run": "parity_data_validation",
        "passed": True,
        "token_map": {
            "bit_zero": BIT_ZERO,
            "bit_one": BIT_ONE,
            "query": QUERY,
            "answer_start": ANSWER_START,
            "ignore": IGNORE,
        },
        "tests": tests,
        "sample_inputs": sample_x.tolist(),
        "sample_labels": sample_y.tolist(),
    }

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("determinism: PASS")
    print("independent seeds: PASS")
    print("query and masking: PASS")
    print("answer recomputation: PASS")
    print("answer leakage: PASS")
    print("shape and difficulty sweep: PASS")
    print("==============================")
    print("PARITY DATA VALIDATION: PASS")
    print("saved:", RESULT)


if __name__ == "__main__":
    main()
