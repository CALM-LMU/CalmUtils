"""
Shared convolutional building blocks (pure PyTorch).

ConvBlock is the canonical block used by the models in calmutils.nn (e.g. UNet);
its architecture is described by ConvBlockConfig, so new options are additive
fields with defaults and old configs keep working.
"""

import math
import re
import warnings
from dataclasses import dataclass, asdict

import torch.nn as nn

__all__ = ["ConvBlockConfig", "ConvBlock", "check_dimensionality", "make_activation", "make_norm",
           "remap_legacy_state_dict", "maybe_remap_legacy_state_dict"]

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

    def load_state_dict(self, state_dict, strict=True):
        # accept checkpoints saved with a legacy ConvBlock layout
        return super().load_state_dict(maybe_remap_legacy_state_dict(state_dict), strict=strict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Remap legacy keys IN PLACE: torch's load_state_dict recurses into submodules with
        # dicts filtered from this one, so a remapped copy would not reach deeper layers.
        # (Nested frames receive freshly filtered copies, so in-place mutation is safe here.)
        legacy = {k: v for k, v in state_dict.items() if k.startswith(prefix) and _is_legacy_key(k)}
        if legacy:
            remapped = remap_legacy_state_dict(legacy, context=state_dict)
            for k in legacy:
                del state_dict[k]
            state_dict.update(remapped)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        x1 = self.first(x)
        out = self.rest(x1)
        if self.config.residual == "inner":
            out = out + x1
        elif self.config.residual == "input":
            out = out + self.skip(x)
        return out


# ---------------------------------------------------------------------------
# Legacy state-dict remapping (ConvBlock layout history)
# ---------------------------------------------------------------------------

_LEGACY_FLAT_REST = re.compile(r"\.rest\.\d+\.(weight|bias|running_mean|running_var|num_batches_tracked)$")


def _is_legacy_key(key):
    return ".steps." in key or bool(_LEGACY_FLAT_REST.search(key))


def maybe_remap_legacy_state_dict(state_dict, prefix=""):
    """Return a copy of state_dict with legacy ConvBlock keys remapped (only keys under prefix)."""
    legacy = {k: v for k, v in state_dict.items() if k.startswith(prefix) and _is_legacy_key(k)}
    if not legacy:
        return state_dict
    remapped = remap_legacy_state_dict(legacy, context=state_dict)
    return {k: v for k, v in state_dict.items() if not (k.startswith(prefix) and _is_legacy_key(k))} | remapped


def remap_legacy_state_dict(state_dict, context=None):
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
    context = state_dict if context is None else context  # probe source; may be larger than state_dict

    def sublayer_len(prefix):
        # batch norm adds a parameterized layer at position 1 of the first sublayer
        for probe in (f"{prefix}.steps.1.weight", f"{prefix}.first.1.weight"):
            if probe in context:
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
