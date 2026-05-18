"""CUDA / PyTorch sanity check.

Run after `conda env create -f environment.yml` (or `pip install -r requirements.txt`)
and before any training. Expected output on the dev box (RTX 3060 Ti, driver >= 550):

    torch=2.5.1 cuda_available=True cuda=12.4 device=NVIDIA GeForce RTX 3060 Ti
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch not installed; activate the wind-d env first", file=sys.stderr)
        return 1

    available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if available else "N/A"
    device_name = torch.cuda.get_device_name(0) if available else "cpu-only"
    print(
        f"torch={torch.__version__} cuda_available={available} "
        f"cuda={cuda_version} device={device_name}"
    )
    return 0 if available else 2


if __name__ == "__main__":
    raise SystemExit(main())
