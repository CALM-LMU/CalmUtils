"""
Optional Lightning wrappers for the calmutils.nn models.
Requires the 'lightning' package (optional dependency); without it, the wrapper
classes are stubs that raise an informative ImportError on instantiation.
"""

from .unet import UNet, UNetConfig
from .blocks import ConvBlockConfig

try:
    from lightning import pytorch as L
    _HAS_LIGHTNING = True
except ImportError:  # lightning is an optional dependency
    L = None
    _HAS_LIGHTNING = False

__all__ = ["LightningUNet"]


if _HAS_LIGHTNING:

    class LightningUNet(L.LightningModule):
        """
        Thin wrapper of the PyTorch UNet for lightning.
        Only inference is implemented; for training, subclass this.

        out_channels / in_channels are the primary names (matching UNet); the legacy
        keywords n_classes / input_channels are still accepted as aliases.
        Alternatively pass a full UNetConfig via config=.
        """

        def __init__(self, out_channels=None, intermediate_channels=None, in_channels=None, unet_kwargs=None, config=None, **kwargs):
            super().__init__()

            # legacy keyword aliases (also how old checkpoints' hparams arrive)
            n_classes = kwargs.pop("n_classes", None)
            input_channels = kwargs.pop("input_channels", None)
            if out_channels is None:
                out_channels = n_classes
            elif n_classes is not None and n_classes != out_channels:
                raise ValueError(f"Conflicting out_channels={out_channels} and n_classes={n_classes}.")
            if in_channels is None:
                in_channels = input_channels if input_channels is not None else 1
            elif input_channels is not None and input_channels != in_channels:
                raise ValueError(f"Conflicting in_channels={in_channels} and input_channels={input_channels}.")
            if out_channels is None or intermediate_channels is None:
                raise TypeError("LightningUNet: out_channels and intermediate_channels are required.")

            # on load_from_checkpoint, all hparams (incl. stored unet_config) arrive as kwargs
            stored_config = kwargs.pop("unet_config", None)
            if config is None and stored_config is not None:
                config = stored_config

            if config is None:
                unet_kwargs = dict(unet_kwargs or {})
                # legacy defaults of this wrapper
                residual = unet_kwargs.pop("residual_conv_blocks", True)
                batch_norm = unet_kwargs.pop("batch_norm", True)
                n_convs = unet_kwargs.pop("n_convs_per_block", 3)
                dimensionality = unet_kwargs.pop("dimensionality", 2)
                block = ConvBlockConfig(n_convs=n_convs, residual=residual,
                                        norm="batch" if batch_norm else "none", **unet_kwargs)
                config = UNetConfig(in_channels=in_channels, out_channels=out_channels,
                                    intermediate_channels=list(intermediate_channels),
                                    dimensionality=dimensionality, block=block)
            elif isinstance(config, dict):
                config = UNetConfig.from_dict(config)

            self.unet = UNet(config)
            self.save_hyperparameters(ignore=["config"])
            # keep the legacy hparam keys available for backward compatibility, and
            # store the full architecture config in the checkpoint hyperparameters
            self.hparams["n_classes"] = out_channels
            self.hparams["input_channels"] = in_channels
            self.hparams["unet_config"] = config.to_dict()

        @property
        def config(self):
            return self.unet.config

        def forward(self, x):
            # if given a DataLoader batch (collection of tensors) instead of just input, use the first
            if isinstance(x, (tuple, list)):
                x = x[0]
            return self.unet(x)

else:

    class LightningUNet:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("LightningUNet requires the 'lightning' package (optional dependency).")
