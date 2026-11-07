"""Portable PyTorch device selection for local experiment scripts."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Use Apple MPS when available, otherwise CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def backend_name(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "mps":
        return f"Apple MPS; PyTorch {torch.__version__}"
    return f"CPU; PyTorch {torch.__version__}"
