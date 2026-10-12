"""공통 seed 고정 유틸. 세 task 모두 이걸로 재현성 확보."""
import os
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Turing(sm_75)에서도 결정론적 conv/matmul 강제 (속도 약간 손해, 재현성 우선)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
