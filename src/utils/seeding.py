"""Global determinism helper.

Lifted from `ARCHITECTURE.md` §10. Call `set_global_seed(42)` at the very top of
any training/inference entry-point, *before* importing model code.

Notes:
- `CUBLAS_WORKSPACE_CONFIG=":4096:8"` is required for `torch.use_deterministic_algorithms`
  on CUDA >= 10.2.
- `torch.use_deterministic_algorithms(..., warn_only=True)` lets ops without a
  deterministic kernel fall back to non-deterministic — strict mode would crash
  some PyTorch Forecasting layers.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) deterministically.

    Imports of `numpy` and `torch` are deferred so this module is importable
    in environments where they are not yet installed (e.g., during repo bootstrap).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass
