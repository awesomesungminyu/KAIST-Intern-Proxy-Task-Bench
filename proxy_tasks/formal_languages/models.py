"""parity / tomita5 / tomita6 모델 등록 지점.

모델 구현은 상류 Transformer-Formal-Languages(third_party/tfl) 안에 있다. 여기서는 이름을
상류 runner 인자로 바꾸는 표만 둔다. 상류가 지원하지 않는 새 구조는 TFL 소스
(src/components/)에 구현을 추가하고 src/model.py 가 그것을 고르게 패치해야 한다
(docs/ADDING_A_MODEL.md). torch 를 import 하지 않는다 (validation runner 가 읽는다).
"""

TFL_MODELS: dict[str, list[str]] = {}


def register_model(name: str, tfl_args: list[str]) -> None:
    if name in TFL_MODELS:
        raise ValueError(f"[formal] model already registered: {name}")
    TFL_MODELS[name] = list(tfl_args)


register_model("lstm", ["-model_type", "RNN", "-cell_type", "LSTM", "-hidden_size", "32", "-depth", "1",
                        "-lr", "0.001"])
register_model("san_abs", ["-model_type", "SAN", "-pos_encode", "-d_model", "32", "-depth", "2", "-heads", "4",
                           "-lr", "0.001"])
register_model("san_mask", ["-model_type", "SAN", "-no-pos_encode", "-d_model", "32", "-depth", "2",
                            "-heads", "4", "-lr", "0.001"])
