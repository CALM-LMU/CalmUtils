"""Generic PyTorch utilities (model-agnostic)."""

import warnings

import torch

__all__ = ["compile_if_possible"]


def compile_if_possible(model, **compile_kwargs):
    """
    Generic torch.compile wrapper, e.g. for inference on any model.
    If compilation fails, falls back to the eager model with a warning instead of raising.
    """
    try:
        return torch.compile(model, **compile_kwargs)
    except Exception as e:
        warnings.warn(f"compile_if_possible: torch.compile failed ({e}); using eager model.")
        return model
