"""
PyTorch models and building blocks (calmutils.nn).

Requires torch; lightning is only needed for the lightning_wrappers module.
Both are optional dependencies of CalmUtils (install with pip install CalmUtils[ml]).
Exports are lazy, so importing calmutils.nn without torch installed raises a clear
ImportError only when a torch-dependent name is actually accessed.
"""

import importlib

_LAZY_EXPORTS = {
    "ConvBlock": "blocks",
    "ConvBlockConfig": "blocks",
    "UNet": "unet",
    "UNetConfig": "unet",
    "save_checkpoint": "unet",
    "load_unet_checkpoint": "unet",
    "load_state_dict_flexible": "unet",
    "remap_legacy_state_dict": "unet",
    "compile_if_possible": "utils",
    "focal_loss": "losses",
    "cosine_distance_loss": "losses",
    "negative_cosine_similarity_loss": "losses",
    "LightningUNet": "lightning_wrappers",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        try:
            module = importlib.import_module(f".{_LAZY_EXPORTS[name]}", __name__)
        except ImportError as e:
            raise ImportError(
                f"calmutils.nn.{name} requires optional dependencies "
                f"(pip install CalmUtils[ml]). Original error: {e}"
            ) from e
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
