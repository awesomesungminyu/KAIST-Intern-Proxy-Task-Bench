"""RegBench: 랜덤 DFA -> 균등 확률 PFA 변환 -> 시퀀스 샘플링.

파일 다운로드가 없는 생성형 데이터셋. 두 논문의 하이퍼파라미터가 다르므로
둘 다 프리셋으로 남겨둔다:

- Akyürek et al. 2024 "In-Context Language Learning" 원논문 계열
  (TVP 논문 2501.09240이 "following Akyürek et al. 2024"라고 인용한 수치):
  vocab_size=40, max_num_states=20, max_out_degree=10
- 공식 구현 berlino/seq_icl 의 기본 config(icl_dfa.yaml) 수치:
  vocab_size=20, max_num_states=12, max_out_degree=4

어느 쪽을 쓸지는 팀 방침에 맞춰 PFAConfig 생성 시 골라 쓰면 됨.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

PAD = 0
SEP = 1  # 예시 문자열 구분자 "|"
NUM_SPECIAL_TOKENS = 2  # 실제 vocab 심볼은 이 뒤부터 id를 받음


@dataclass
class PFAConfig:
    vocab_size: int = 40
    max_num_states: int = 20
    max_out_degree: int = 10
    min_string_len: int = 3
    max_string_len: int = 10

    @staticmethod
    def paper_akyurek2024() -> "PFAConfig":
        return PFAConfig(vocab_size=40, max_num_states=20, max_out_degree=10)

    @staticmethod
    def repo_default() -> "PFAConfig":
        return PFAConfig(vocab_size=20, max_num_states=12, max_out_degree=4)


DFA = Dict[int, Dict[int, int]]  # state -> {symbol_id: next_state}


def sample_dfa(cfg: PFAConfig, rng: random.Random) -> DFA:
    """랜덤 DFA 생성. 각 상태는 1~max_out_degree개의 서로 다른 심볼로
    나가는 edge를 가짐 (결정적: 같은 상태에서 같은 심볼로 두 개의 edge가 나갈 수 없음).
    """
    num_states = rng.randint(2, cfg.max_num_states)
    dfa: DFA = {}
    for s in range(num_states):
        out_degree = rng.randint(1, min(cfg.max_out_degree, cfg.vocab_size))
        symbols = rng.sample(range(cfg.vocab_size), out_degree)
        dfa[s] = {sym: rng.randrange(num_states) for sym in symbols}
    return dfa


def dfa_to_pfa(dfa: DFA) -> Dict[int, Dict[int, Tuple[int, float]]]:
    """각 상태의 나가는 edge에 균등 확률을 부여. state -> {symbol: (next_state, prob)}"""
    pfa = {}
    for state, edges in dfa.items():
        p = 1.0 / len(edges)
        pfa[state] = {sym: (nxt, p) for sym, nxt in edges.items()}
    return pfa


def sample_string(
    pfa: Dict[int, Dict[int, Tuple[int, float]]],
    cfg: PFAConfig,
    rng: random.Random,
    start_state: int = 0,
) -> List[int]:
    """PFA에서 시퀀스 하나 샘플링. 나갈 edge가 없는 상태(dead-end)에 도달하거나
    max_string_len에 닿으면 종료. 심볼 id는 NUM_SPECIAL_TOKENS만큼 offset."""
    state = start_state if start_state in pfa else rng.choice(list(pfa.keys()))
    out: List[int] = []
    target_len = rng.randint(cfg.min_string_len, cfg.max_string_len)
    for _ in range(target_len):
        edges = pfa.get(state)
        if not edges:
            break
        symbols, weights = zip(*[(sym, w) for sym, (_, w) in edges.items()])
        sym = rng.choices(symbols, weights=weights, k=1)[0]
        nxt, _ = edges[sym]
        out.append(sym + NUM_SPECIAL_TOKENS)
        state = nxt
    return out


def build_icl_sequence(
    pfa: Dict[int, Dict[int, Tuple[int, float]]],
    cfg: PFAConfig,
    rng: random.Random,
    num_examples: int,
    max_total_len: int,
) -> List[int]:
    """같은 PFA에서 뽑은 여러 문자열을 SEP으로 이어 붙여 하나의 in-context 시퀀스로
    구성. 언어모델 loss는 이 전체 시퀀스에 대한 next-token prediction."""
    seq: List[int] = []
    for i in range(num_examples):
        s = sample_string(pfa, cfg, rng)
        if not s:
            continue
        if seq:
            seq.append(SEP)
        seq.extend(s)
        if len(seq) >= max_total_len:
            break
    return seq[:max_total_len]


def vocab_size_with_specials(cfg: PFAConfig) -> int:
    return cfg.vocab_size + NUM_SPECIAL_TOKENS
