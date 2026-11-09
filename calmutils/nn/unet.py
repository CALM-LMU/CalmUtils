"""
Flexible U-Net implementation in pure PyTorch.

Architecture is described entirely by the UNetConfig dataclass, so new options are
additive fields with defaults and old configs keep working. Checkpoints saved via
save_checkpoint() store the config next to the weights, so models can be
reconstructed independently of the current class layout.
"""

import dataclasses
import re
import warnings
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, ConvBlockConfig, check_dimensionality, conv_classes, conv_transpose_classes

__all__ = [
    "UNetConfig",
    "UNet",
    "save_checkpoint",
    "load_unet_checkpoint",
    "load_state_dict_flexible",
    "remap_legacy_state_dict",
]


@dataclass
class UNetConfig:
    """
    Configuration of a UNet. schema_version is bumped on incompatible changes
    (v3 introduced the residual mode enum); loading a config/checkpoint with a
    newer schema_version than this code warns loudly.
    """
    schema_version: int = 3
    in_channels: int = 1
    out_channels: int = 1
    intermediate_channels: list = field(default_factory=lambda: [64, 128, 128])
    dimensionality: int = 2
    block: ConvBlockConfig = field(default_factory=ConvBlockConfig)
    upsampling: str = "interpolate"  # "interpolate" (1x1 conv + interpolate) | "transpose" (transposed conv)
    interpolate_mode: str = "nearest"  # mode for F.interpolate when upsampling="interpolate"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        stored_version = d.get("schema_version", 1)
        if isinstance(stored_version, int) and stored_version > cls.schema_version:
            warnings.warn(f"UNetConfig: config was written with schema_version {stored_version}, "
                          f"but this code supports up to {cls.schema_version}; architecture may not load correctly.")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - known
        if unknown:
            warnings.warn(f"UNetConfig: ignoring unknown keys {sorted(unknown)} (config written by a newer version?)")
        if "block" in d and isinstance(d["block"], dict):
            d["block"] = ConvBlockConfig.from_dict(d["block"])
        return cls(**{k: v for k, v in d.items() if k in known})


class UNet(nn.Module):
    """
    U-Net built from a UNetConfig.

    Can also be constructed with the legacy positional signature
    UNet(in_channels, intermediate_channels, out_channels, dimensionality=2,
         residual_conv_blocks=False, n_convs_per_block=3, batch_norm=False),
    which is mapped onto an equivalent UNetConfig.
    """

    def __init__(self, config=None, *legacy_args, dimensionality=2,
                 residual_conv_blocks=False, n_convs_per_block=3, batch_norm=False, **kwargs):
        super().__init__()

        if kwargs:
            raise TypeError(f"UNet: unknown keyword arguments {sorted(kwargs)}")

        if config is not None and isinstance(config, (UNetConfig, dict)):
            if isinstance(config, dict):
                config = UNetConfig.from_dict(config)
        else:
            # legacy positional style: UNet(in_channels, intermediate_channels, out_channels, [dimensionality])
            args = (config, *legacy_args) if config is not None else legacy_args
            if len(args) < 3:
                raise TypeError("UNet: pass a UNetConfig/dict, or legacy positional "
                                "(in_channels, intermediate_channels, out_channels, ...)")
            in_channels, intermediate_channels, out_channels = args[0], args[1], args[2]
            if len(args) > 3:
                dimensionality = args[3]
            block = ConvBlockConfig(n_convs=n_convs_per_block, residual=residual_conv_blocks,
                                    norm="batch" if batch_norm else "none")
            config = UNetConfig(in_channels=in_channels, out_channels=out_channels,
                                intermediate_channels=list(intermediate_channels),
                                dimensionality=dimensionality, block=block)

        self.config = config
        dimensionality = config.dimensionality
        check_dimensionality(dimensionality)
        conv_class = conv_classes(dimensionality)
        self.dimensionality = dimensionality

        intermediate = list(config.intermediate_channels)
        n_encoder_blocks = len(intermediate)
        if n_encoder_blocks < 2:
            raise ValueError("UNet: intermediate_channels must contain at least 2 entries.")

        block = config.block

        # encoder: first and last block do not pool, intermediate blocks downsample via stride
        encoder = {}
        for i, n_channels in enumerate(intermediate):
            stride_1st = 2 if 0 < i < n_encoder_blocks - 1 else 1
            block_i = ConvBlock(config.in_channels if i == 0 else intermediate[i - 1], n_channels,
                                dataclasses.replace(block, stride_1st=stride_1st), dimensionality)
            encoder[f'encoder_block_{i}'] = block_i
        self.encoder = nn.ModuleDict(encoder)

        # decoder: one step less than encoder (last encoder output is the bottom of the U)
        decoder = {}
        for i, n_channels in enumerate(reversed(intermediate[:-1])):
            # input channels are twice the normal size, as we concatenate skip connections
            decoder[f'decoder_block_{i}'] = ConvBlock(n_channels * 2, n_channels, block, dimensionality)
        self.decoder = nn.ModuleDict(decoder)

        # upsampling: 1x1 conv + interpolate, or transposed convolution
        if config.upsampling == "interpolate":
            self.up_convs = nn.ModuleList(
                [conv_class(intermediate[n], intermediate[n - 1], 1, padding_mode=block.padding_mode)
                 for n in range(n_encoder_blocks - 1, 0, -1)])
        elif config.upsampling == "transpose":
            self.up_convs = nn.ModuleList(
                [conv_transpose_classes(dimensionality)(intermediate[n], intermediate[n - 1], kernel_size=2, stride=2)
                 for n in range(n_encoder_blocks - 1, 0, -1)])
        else:
            raise ValueError(f"Unknown upsampling '{config.upsampling}'.")

        # final 1x1 convolution to get output channel number
        self.out_conv = conv_class(intermediate[0], config.out_channels, 1, padding_mode=block.padding_mode)

    def forward(self, x):
        # apply encoder conv blocks, keep intermediate outputs for skip connections
        encoder_intermediates = []
        for enc in self.encoder.values():
            x = enc(x)
            encoder_intermediates.append(x)

        for i, (upc, dec) in enumerate(zip(self.up_convs, self.decoder.values())):
            # skip connection from corresponding encoder layer
            x2 = encoder_intermediates[::-1][i + 1]
            x = upc(x)
            if self.config.upsampling == "interpolate":
                # interpolate to skip-connection size (robust to odd input sizes)
                x = F.interpolate(x, size=x2.shape[-self.dimensionality:], mode=self.config.interpolate_mode)
            else:
                # transposed conv may be off by one for odd sizes: center-crop or pad to match
                x = _match_spatial_size(x, x2.shape[-self.dimensionality:])
            x = torch.cat([x, x2], 1)
            x = dec(x)

        return self.out_conv(x)


def _match_spatial_size(x, target_shape):
    """Center-crop (or zero-pad) the spatial dimensions of x to target_shape."""
    slices = [slice(None), slice(None)]
    pads = []
    for cur, tgt in zip(x.shape[2:], target_shape):
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
        else:
            before = (tgt - cur) // 2
            slices.append(slice(None))
            pads += [before, tgt - cur - before]
    if any(s != slice(None) for s in slices[2:]):
        x = x[tuple(slices)]
    if pads:
        # F.pad expects padding for last dim first
        x = F.pad(x, pads)
    return x


# ---------------------------------------------------------------------------
# Checkpoint / state-dict compatibility helpers
# ---------------------------------------------------------------------------

def remap_legacy_state_dict(state_dict):
    """
    Remap a state dict from a legacy ConvBlock layout to the current one.

    Legacy layouts handled:
    - schema_version 1: a single flat 'steps' Sequential of sublayers
      (conv, [batch norm], activation, ...) per block.
    - intermediate (uncommitted refactor): 'first' sublayer + flat 'rest' Sequential.

    The current layout uses 'first' (one sublayer) and 'rest' (a Sequential of
    sublayers, each itself a Sequential), so flat indices beyond the first sublayer
    are decomposed into (sublayer index, position within sublayer).
    """
    flat_rest = re.compile(r"^(.*\.rest)\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$")

    def sublayer_len(prefix):
        # batch norm adds a parameterized layer at position 1 of the first sublayer
        for probe in (f"{prefix}.steps.1.weight", f"{prefix}.first.1.weight"):
            if probe in state_dict:
                return 3
        return 2

    remapped = {}
    for key, value in state_dict.items():
        if ".steps." in key:
            prefix, rest = key.rsplit(".steps.", 1)
            idx_str, suffix = rest.split(".", 1)
            idx = int(idx_str)
            split = sublayer_len(prefix)
            if idx < split:
                remapped[f"{prefix}.first.{idx}.{suffix}"] = value
            else:
                s, p = divmod(idx - split, split)
                remapped[f"{prefix}.rest.{s}.{p}.{suffix}"] = value
        elif flat_rest.match(key):
            m = flat_rest.match(key)
            prefix, idx, suffix = m.group(1), int(m.group(2)), m.group(3)
            block_prefix = prefix[:-len(".rest")]
            s, p = divmod(idx, sublayer_len(block_prefix))
            remapped[f"{prefix}.{s}.{p}.{suffix}"] = value
        else:
            remapped[key] = value
    return remapped


def load_state_dict_flexible(model, state_dict, verbose=True):
    """
    Load a state dict, falling back to the legacy-key remap if needed.
    Never fully silent: returns (missing_keys, unexpected_keys) and warns if non-empty.
    """
    try:
        model.load_state_dict(state_dict, strict=True)
        return [], []
    except RuntimeError:
        pass
    flat_rest = re.compile(r"\.rest\.\d+\.(weight|bias|running_mean|running_var|num_batches_tracked)$")
    if any(".steps." in k or flat_rest.search(k) for k in state_dict):
        candidate = remap_legacy_state_dict(state_dict)
    else:
        candidate = state_dict
    result = model.load_state_dict(candidate, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if verbose and (missing or unexpected):
        warnings.warn(f"load_state_dict_flexible: missing keys: {missing}; unexpected keys: {unexpected}")
    return missing, unexpected


def save_checkpoint(model, path, extra_metadata=None):
    """Save config + weights together, so the model can be rebuilt independently of class layout."""
    ckpt = {
        "format": "unet-checkpoint",
        "schema_version": model.config.schema_version,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
    }
    if extra_metadata:
        ckpt["extra"] = extra_metadata
    torch.save(ckpt, path)


def load_unet_checkpoint(path, map_location="cpu", config=None, verbose=True):
    """Load a checkpoint written by save_checkpoint(). If the file is a plain state dict, a config must be given."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "config" in ckpt and "state_dict" in ckpt:
        config = UNetConfig.from_dict(ckpt["config"])
        state_dict = ckpt["state_dict"]
    else:
        if config is None:
            raise ValueError("Checkpoint has no stored config; pass config=UNetConfig(...) explicitly.")
        if isinstance(config, dict):
            config = UNetConfig.from_dict(config)
        state_dict = ckpt
    model = UNet(config)
    load_state_dict_flexible(model, state_dict, verbose=verbose)
    return model.to(map_location)
