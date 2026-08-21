"""Render tensors as metadata while recovering a masked worker exception."""

import os


if os.environ.get("QWEN38_SAFE_TENSOR_REPR") == "1":
    try:
        import torch
    except ImportError:
        pass
    else:

        def _metadata_only(tensor: torch.Tensor) -> str:
            try:
                return (
                    "Tensor("
                    f"shape={tuple(tensor.shape)}, "
                    f"dtype={tensor.dtype}, "
                    f"device={tensor.device}, "
                    f"layout={tensor.layout}, "
                    f"requires_grad={tensor.requires_grad}"
                    ")"
                )
            except Exception:
                return "Tensor(metadata_unavailable)"

        torch.Tensor.__repr__ = _metadata_only
        torch.Tensor.__str__ = _metadata_only
