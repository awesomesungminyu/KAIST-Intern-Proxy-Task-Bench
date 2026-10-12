"""CMUDict 다운로드 + 파싱 + split 생성.

출처: cmusphinx/cmudict (활발히 유지보수되는 정본).
    https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
    (실측 3,618,488 bytes, 2026-09-28 확인)

주의: 이 파일은 Yao & Zweig(2015)/Yolchuyeva et al.(2019) 논문이 쓴
cmudict-0.7b(고정 버전)와 표제어 집합이 다르다. 논문 split 파일을
안정적으로 구할 수 없어서, 여기서는 **결정론적 랜덤 split(90/5/5, seed 고정)**을
쓴다. "논문과 100% 동일 재현"이 아니라 "동등한 난이도의 재구성"이라는 점을
README에 명시했으니 참고.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path
from typing import List, Tuple

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
DEFAULT_OUT = Path(__file__).parent / "data"


def download_cmudict(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[g2p] already downloaded: {out_path} ({out_path.stat().st_size} bytes)")
        return out_path
    print(f"[g2p] downloading {CMUDICT_URL} ...")
    urllib.request.urlretrieve(CMUDICT_URL, out_path)
    size = out_path.stat().st_size
    print(f"[g2p] downloaded {size} bytes -> {out_path}")
    return out_path


def parse_cmudict(path: Path) -> List[Tuple[str, List[str]]]:
    """(grapheme_word, phoneme_list) 튜플 리스트로 파싱.
    - 주석(;;;)과 빈 줄 제외
    - 알파벳/아포스트로피만으로 이뤄진 표제어만 사용 (숫자·기호 포함 단어 제외)
    - 변이 발음 표기 WORD(2) 형태는 기본형 WORD로 정규화하되, 발음은 별개 예시로 유지
    """
    entries = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()  # 인라인 주석(예: "# place, danish") 제거
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word, phones = parts[0], parts[1:]
            # WORD(2) -> WORD 로 정규화 (발음 variant 표시 제거)
            if "(" in word and word.endswith(")"):
                word = word[: word.index("(")]
            if not all(c.isalpha() or c == "'" for c in word):
                continue
            entries.append((word.upper(), phones))
    return entries


def split_entries(
    entries: List[Tuple[str, List[str]]], seed: int, ratios=(0.90, 0.05, 0.05)
):
    """단어(그래프) 단위로 split — 같은 단어의 서로 다른 발음 variant가
    train/test에 걸쳐 누출되지 않도록 단어 기준으로 나눔."""
    words = sorted(set(w for w, _ in entries))
    rng = random.Random(seed)
    rng.shuffle(words)
    n = len(words)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_words = set(words[:n_train])
    val_words = set(words[n_train : n_train + n_val])
    test_words = set(words[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for w, phones in entries:
        if w in train_words:
            splits["train"].append((w, phones))
        elif w in val_words:
            splits["val"].append((w, phones))
        else:
            splits["test"].append((w, phones))
    return splits


def build_vocab(entries: List[Tuple[str, List[str]]]):
    chars = sorted(set(c for w, _ in entries for c in w))
    phones = sorted(set(p for _, ph in entries for p in ph))
    # 0=PAD, 1=BOS, 2=EOS 예약
    char_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for c in chars:
        char_vocab[c] = len(char_vocab)
    phone_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for p in phones:
        phone_vocab[p] = len(phone_vocab)
    return char_vocab, phone_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_path = out_dir / "cmudict.dict"
    download_cmudict(raw_path)

    entries = parse_cmudict(raw_path)
    print(f"[g2p] parsed {len(entries)} pronunciation entries "
          f"({len(set(w for w, _ in entries))} unique words)")

    splits = split_entries(entries, seed=args.seed)
    for name, items in splits.items():
        print(f"[g2p] split {name}: {len(items)} entries")

    char_vocab, phone_vocab = build_vocab(entries)
    print(f"[g2p] char vocab size={len(char_vocab)}  phone vocab size={len(phone_vocab)}")

    with open(out_dir / "vocab.json", "w") as f:
        json.dump({"char": char_vocab, "phone": phone_vocab}, f, ensure_ascii=False, indent=2)
    for name, items in splits.items():
        with open(out_dir / f"{name}.tsv", "w", encoding="utf-8") as f:
            for w, phones in items:
                f.write(f"{w}\t{' '.join(phones)}\n")
    print(f"[g2p] wrote split files + vocab.json -> {out_dir}")


if __name__ == "__main__":
    main()
