"""
Shared convolutional building blocks (pure PyTorch).

ConvBlock is the canonical block used by the models in calmutils.nn (e.g. UNet);
its architecture is described by ConvBlockConfig, so new options are additive
fields with defaults and old configs keep working.
"""

import math
import warnings
from dataclasses import dataclass, asdict

import torch.nn as nn

__all__ = ["ConvBlockConfig", "ConvBlock", "check_dimensionality", "make_activation", "make_norm"]

_CONV_CLASSES = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_BATCHNORM_CLASSES = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
_DROPOUT_CLASSES = {1: nn.Dropout1d, 2: nn.Dropout2d, 3: nn.Dropout3d}


def check_dimensionality(dimensionality):
    if dimensionality not in (1, 2, 3):
        raise ValueError("Only 1D, 2D, 3D is supported.")


def make_activation(name, leaky_relu_slope=0.01):
    name = name.lower()
    if name == "leaky_relu":
        return nn.LeakyReLU(leaky_relu_slope, inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown activation '{name}'.")


def make_norm(name, num_features, dimensionality, n_norm_groups=8):
    """Return a norm layer, or None for 'none'. GroupNorm groups are reduced to a divisor of num_features."""
    name = name.lower()
    if name == "none":
        return None
    if name == "batch":
        return _BATCHNORM_CLASSES[dimensionality](num_features)
    if name == "group":
        return nn.GroupNorm(math.gcd(n_norm_groups, num_features), num_features)
    raise ValueError(f"Unknown norm '{name}'.")


def conv_classes(dimensionality):
    check_dimensionality(dimensionality)
    return _CONV_CLASSES[dimensionality]


def conv_transpose_classes(dimensionality):
    check_dimensionality(dimensionality)
    return {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[dimensionality]


def dropout_classes(dimensionality):
    check_dimensionality(dimensionality)
    return _DROPOUT_CLASSES[dimensionality]


@dataclass
class ConvBlockConfig:
    """Configuration of a ConvBlock. All fields have defaults; new fields may be added over time."""
    n_convs: int = 3
    kernel_size: int = 3
    stride_1st: int = 1
    residual: str = "none"  # "none" | "inner" (add first-sublayer output) | "input" (true residual, projection shortcut)
    activation: str = "leaky_relu"
    leaky_relu_slope: float = 0.01
    norm: str = "none"  # "none" | "batch" | "group"
    n_norm_groups: int = 8  # used for norm="group"
    groups: int = 1  # convolution groups (e.g. >1 for grouped/depthwise convs)
    padding_mode: str = "zeros"
    dropout: float = 0.0

    def __post_init__(self):
        # legacy bool values: True meant "add first-sublayer output" ("inner")
        if isinstance(self.residual, bool):
            self.residual = "inner" if self.residual else "none"
        if self.residual not in ("none", "inner", "input"):
            raise ValueError(f"Unknown residual mode '{self.residual}' (expected 'none', 'inner' or 'input').")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - known
        if unknown:
            warnings.warn(f"ConvBlockConfig: ignoring unknown keys {sorted(unknown)} (config written by a newer version?)")
        return cls(**{k: v for k, v in d.items() if k in known})


class ConvBlock(nn.Module):
    """
    Multiple conv -> [norm] -> activation -> [dropout] sublayers.

    The first conv maps from in_channels to out_channels and may have stride > 1 to achieve pooling.
    All other convs retain the spatial shape of the first intermediate output.

    Residual modes:
    - "none":  plain feed-forward block.
    - "inner": output of the first sublayer is added to the block output (shapes always
      match, also with stride_1st > 1, since both sides are computed after the first conv).
    - "input": true residual — the block input is added to the block output; if shapes
      differ (channel change and/or stride), a 1x1 projection conv (+ norm) is used as shortcut.
    """

    def __init__(self, in_channels, out_channels, config: ConvBlockConfig, dimensionality=2):
        super().__init__()
        check_dimensionality(dimensionality)
        conv_class = _CONV_CLASSES[dimensionality]
        self.config = config
        padding = config.kernel_size // 2

        def make_sublayer(c_in, c_out, stride):
            layers = [conv_class(c_in, c_out, config.kernel_size, stride=stride, padding=padding,
                                 groups=config.groups, padding_mode=config.padding_mode)]
            norm = make_norm(config.norm, c_out, dimensionality, config.n_norm_groups)
            if norm is not None:
                layers.append(norm)
            layers.append(make_activation(config.activation, config.leaky_relu_slope))
            if config.dropout > 0:
                layers.append(_DROPOUT_CLASSES[dimensionality](config.dropout))
            return nn.Sequential(*layers)

        self.first = make_sublayer(in_channels, out_channels, config.stride_1st)
        rest_layers = [make_sublayer(out_channels, out_channels, 1) for _ in range(config.n_convs - 1)]
        self.rest = nn.Sequential(*rest_layers) if rest_layers else nn.Identity()

        # projection shortcut for residual="input" (only needed when shapes differ)
        self.skip = nn.Identity()
        if config.residual == "input" and (in_channels != out_channels or config.stride_1st != 1):
            skip_layers = [conv_class(in_channels, out_channels, 1, stride=config.stride_1st,
                                      padding_mode=config.padding_mode)]
            skip_norm = make_norm(config.norm, out_channels, dimensionality, config.n_norm_groups)
            if skip_norm is not None:
                skip_layers.append(skip_norm)
            self.skip = nn.Sequential(*skip_layers)

    def forward(self, x):
        x1 = self.first(x)
        out = self.rest(x1)
        if self.config.residual == "inner":
            out = out + x1
        elif self.config.residual == "input":
            out = out + self.skip(x)
        return out
