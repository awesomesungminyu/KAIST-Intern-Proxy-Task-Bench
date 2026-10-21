import json
from pathlib import Path

import torch


WRITE = 1
READ = 2
IGNORE = 3
BIT_ZERO = 4
BIT_ONE = 5
MASK = -100


def make_dataset(
    num_examples,
    num_instructions,
    ignore_probability,
    seed,
):
    generator = torch.Generator().manual_seed(seed)

    other_probability = (1.0 - ignore_probability) / 2.0
    probabilities = torch.tensor([
        other_probability,
        other_probability,
        ignore_probability,
    ])

    operations = torch.multinomial(
        probabilities,
        num_examples * num_instructions,
        replacement=True,
        generator=generator,
    ).reshape(num_examples, num_instructions) + WRITE

    # 모든 시퀀스는 초기 상태를 정의하는 write로 시작한다.
    operations[:, 0] = WRITE

    values = torch.randint(
        0,
        2,
        (num_examples, num_instructions),
        generator=generator,
    )

    inputs = torch.empty(
        (num_examples, num_instructions * 2),
        dtype=torch.long,
    )
    labels = torch.full_like(inputs, MASK)

    state = torch.zeros(num_examples, dtype=torch.long)
    last_write = torch.zeros(num_examples, dtype=torch.long)
    read_distances = []

    for step in range(num_instructions):
        operation = operations[:, step]
        value = values[:, step]

        write_mask = operation == WRITE
        read_mask = operation == READ

        state = torch.where(write_mask, value, state)
        last_write = torch.where(
            write_mask,
            torch.full_like(last_write, step),
            last_write,
        )

        # read 뒤의 값은 현재 메모리 상태여야 한다.
        values[:, step] = torch.where(read_mask, state, value)

        operation_position = 2 * step
        value_position = operation_position + 1

        inputs[:, operation_position] = operation
        inputs[:, value_position] = values[:, step] + BIT_ZERO

        # read 토큰 위치에서 다음 값을 예측한다.
        labels[read_mask, operation_position] = (
            state[read_mask] + BIT_ZERO
        )

        if read_mask.any():
            read_distances.extend(
                (step - last_write[read_mask]).tolist()
            )

    return inputs, labels, operations, read_distances


def verify(inputs, labels):
    for row in range(inputs.size(0)):
        state = None

        for position in range(0, inputs.size(1), 2):
            operation = inputs[row, position].item()
            value = inputs[row, position + 1].item() - BIT_ZERO
            target = labels[row, position].item()

            assert value in (0, 1)
            assert labels[row, position + 1].item() == MASK

            if operation == WRITE:
                state = value
                assert target == MASK
            elif operation == IGNORE:
                assert target == MASK
            elif operation == READ:
                assert state is not None
                assert value == state
                assert target == state + BIT_ZERO
            else:
                raise AssertionError("unknown operation token")


def main():
    conditions = [
        ("train_id", 128, 0.80),
        ("dense_ood", 128, 0.10),
        ("sparse_ood", 128, 0.98),
        ("long_sparse_ood", 512, 0.98),
    ]

    records = []

    for name, instructions, ignore_probability in conditions:
        inputs, labels, operations, distances = make_dataset(
            num_examples=100,
            num_instructions=instructions,
            ignore_probability=ignore_probability,
            seed=123,
        )
        same_inputs, same_labels, _, _ = make_dataset(
            num_examples=100,
            num_instructions=instructions,
            ignore_probability=ignore_probability,
            seed=123,
        )
        other_inputs, _, _, _ = make_dataset(
            num_examples=100,
            num_instructions=instructions,
            ignore_probability=ignore_probability,
            seed=456,
        )

        verify(inputs, labels)

        assert torch.equal(inputs, same_inputs)
        assert torch.equal(labels, same_labels)
        assert not torch.equal(inputs, other_inputs)
        assert int((labels != MASK).sum()) == int(
            (operations == READ).sum()
        )

        read_count = int((operations == READ).sum())
        mean_distance = (
            sum(distances) / len(distances) if distances else 0.0
        )
        max_distance = max(distances) if distances else 0

        record = {
            "condition": name,
            "num_examples": 100,
            "num_instructions": instructions,
            "ignore_probability": ignore_probability,
            "read_targets": read_count,
            "mean_distance_since_write": mean_distance,
            "max_distance_since_write": max_distance,
            "passed": True,
        }
        records.append(record)

        print(
            name,
            f"reads={read_count}",
            f"mean_distance={mean_distance:.2f}",
            f"max_distance={max_distance}",
            "PASS",
        )

    output = Path(
        "/data/mqar/results/flipflop_data_validation.json"
    )
    output.write_text(
        json.dumps({
            "run": "flipflop_data_validation",
            "passed": True,
            "format": "operation/value token pairs",
            "label_position": "read operation token",
            "conditions": records,
        }, indent=2),
        encoding="utf-8",
    )

    print("==============================")
    print("FLIP-FLOP DATA VALIDATION: PASS")
    print("saved:", output)


if __name__ == "__main__":
    main()
