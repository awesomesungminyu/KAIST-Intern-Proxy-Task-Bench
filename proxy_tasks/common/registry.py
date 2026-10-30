"""태스크별 모델 등록 지점.

각 태스크의 models.py 가 ModelRegistry 하나를 만들고 `@register_model("이름")` 으로
생성 함수를 등록한다. train.py 의 --model 선택지는 등록된 이름에서 나온다.
따라서 새 아키텍처는 models.py 만 고쳐서 추가한다 (validation runner 는 models*.py
변경을 평가 설정 변경으로 보지 않는다). 생성 함수가 받는 인자는 태스크마다 다르다.
docs/MODELS.md 참조.

torch 를 import 하지 않는다.
"""
from __future__ import annotations


class ModelRegistry:
    def __init__(self, task: str):
        self.task = task
        self._builders = {}

    def register(self, name: str):
        def decorate(builder):
            if name in self._builders:
                raise ValueError(f"[{self.task}] model already registered: {name}")
            self._builders[name] = builder
            return builder
        return decorate

    def names(self) -> list[str]:
        return list(self._builders)

    def build(self, name: str, **kwargs):
        if name not in self._builders:
            raise ValueError(f"[{self.task}] unknown model: {name}; registered: {self.names()}")
        return self._builders[name](**kwargs)
