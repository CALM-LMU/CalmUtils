"""Parity and compatibility tests for calmutils.nn UNet against the legacy snapshot."""
import pytest
import torch

from calmutils.nn.blocks import ConvBlock, ConvBlockConfig
from calmutils.nn.unet import (UNet, UNetConfig, save_checkpoint, load_unet_checkpoint,
                              load_state_dict_flexible, remap_legacy_state_dict)
from calmutils.nn.utils import compile_if_possible
from calmutils.nn.lightning_wrappers import LightningUNet
from unet_legacy_snapshot import UNet as LegacyUNet


def make_pair(seed=0, intermediates=(32, 64, 128), batch_norm=True, residual=True, n_convs=3, dim=2):
    torch.manual_seed(seed)
    legacy = LegacyUNet(3, list(intermediates), 2, dimensionality=dim,
                        residual_conv_blocks=residual, n_convs_per_block=n_convs, batch_norm=batch_norm)
    torch.manual_seed(seed)
    new = UNet(3, list(intermediates), 2, dimensionality=dim,
               residual_conv_blocks=residual, n_convs_per_block=n_convs, batch_norm=batch_norm)
    return legacy.eval(), new.eval()


def test_same_parameters_same_seed():
    legacy, new = make_pair()
    legacy_sd, new_sd = legacy.state_dict(), new.state_dict()
    assert set(legacy_sd) != set(new_sd)  # key layout differs (steps.* vs first/rest.*)
    remapped = remap_legacy_state_dict(legacy_sd)
    assert set(remapped) == set(new_sd)
    for k in new_sd:
        assert torch.equal(remapped[k], new_sd[k]), k


@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("n_convs", [1, 2, 3])
def test_forward_parity(batch_norm, n_convs):
    legacy, new = make_pair(batch_norm=batch_norm, n_convs=n_convs)
    # load legacy weights into new model (exercises the remap)
    missing, unexpected = load_state_dict_flexible(new, legacy.state_dict())
    assert not missing and not unexpected
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(legacy(x), new(x), atol=1e-6)


def test_legacy_state_dict_loads_strictly_after_remap():
    legacy, new = make_pair()
    remapped = remap_legacy_state_dict(legacy.state_dict())
    new.load_state_dict(remapped, strict=True)  # must not raise


def test_legacy_state_dict_loads_via_parent_module():
    # simulates LightningModule.load_from_checkpoint, which bypasses child load_state_dict overrides
    import torch.nn as nn
    legacy, new = make_pair()
    wrapper = nn.Sequential(new)  # parent module holding the UNet
    wrapper.load_state_dict({"0." + k: v for k, v in legacy.state_dict().items()}, strict=True)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(legacy(x), new(x), atol=1e-6)


def test_dims_and_odd_sizes():
    cfg = UNetConfig(in_channels=2, out_channels=3, intermediate_channels=[8, 16, 32], dimensionality=2,
                     block=ConvBlockConfig(n_convs=2, residual=True, norm="batch"))
    model = UNet(cfg).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 2, 96, 96)).shape == (1, 3, 96, 96)
        assert model(torch.randn(1, 2, 95, 97)).shape == (1, 3, 95, 97)  # odd sizes
    cfg1 = UNetConfig(in_channels=1, out_channels=1, intermediate_channels=[8, 16], dimensionality=1)
    m1 = UNet(cfg1).eval()
    with torch.no_grad():
        assert m1(torch.randn(1, 1, 50)).shape == (1, 1, 50)
    cfg3 = UNetConfig(in_channels=1, out_channels=2, intermediate_channels=[4, 8, 16], dimensionality=3)
    m3 = UNet(cfg3).eval()
    with torch.no_grad():
        assert m3(torch.randn(1, 1, 16, 20, 24)).shape == (1, 2, 16, 20, 24)


def test_transpose_upsampling():
    for size in [(64, 64), (63, 65)]:
        cfg = UNetConfig(in_channels=1, out_channels=1, intermediate_channels=[8, 16, 32],
                         upsampling="transpose")
        m = UNet(cfg).eval()
        with torch.no_grad():
            assert m(torch.randn(1, 1, *size)).shape == (1, 1, *size)


def test_config_roundtrip_and_unknown_keys():
    cfg = UNetConfig(in_channels=3, out_channels=2, intermediate_channels=[16, 32],
                     block=ConvBlockConfig(n_convs=2, norm="group", groups=2, residual=True))
    cfg2 = UNetConfig.from_dict(cfg.to_dict())
    assert cfg2 == cfg
    with pytest.warns(UserWarning, match="unknown keys"):
        cfg3 = UNetConfig.from_dict({**cfg.to_dict(), "future_option": 123})
    assert cfg3.in_channels == 3


def test_checkpoint_roundtrip(tmp_path):
    cfg = UNetConfig(in_channels=3, out_channels=1, intermediate_channels=[8, 16, 32],
                     block=ConvBlockConfig(n_convs=2, residual=True, norm="batch"))
    torch.manual_seed(0)
    model = UNet(cfg).eval()
    path = tmp_path / "model.ckpt"
    save_checkpoint(model, path, extra_metadata={"epoch": 5})
    loaded = load_unet_checkpoint(path).eval()  # state dicts don't carry train/eval mode
    x = torch.randn(1, 3, 48, 48)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
    assert loaded.config == cfg


def test_compile_if_possible_fallback(monkeypatch):
    model = UNet(UNetConfig(in_channels=1, out_channels=1, intermediate_channels=[8, 16]))
    monkeypatch.setattr(torch, "compile", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.warns(UserWarning, match="torch.compile failed"):
        out = compile_if_possible(model)
    assert out is model  # falls back to eager model


def test_lightning_wrapper_legacy_signature():
    net = LightningUNet(2, [8, 16, 32], input_channels=3, unet_kwargs={"n_convs_per_block": 2})
    x = torch.randn(1, 3, 32, 32)
    assert net(x).shape == (1, 2, 32, 32)
    # tuple input (dataloader batch) still works
    assert net((x, torch.zeros(1, 32, 32))).shape == (1, 2, 32, 32)
    assert "unet_config" in net.hparams
    assert net.config.out_channels == 2


def test_residual_modes(tmp_path):
    # bool compat: legacy True/False map to inner/none
    assert ConvBlockConfig(residual=True).residual == "inner"
    assert ConvBlockConfig(residual=False).residual == "none"
    assert ConvBlockConfig.from_dict({"residual": True}).residual == "inner"
    with pytest.raises(ValueError, match="Unknown residual"):
        ConvBlockConfig(residual="bogus")
    # strided, channel-changing block: all modes produce matching output shape, incl. odd sizes
    for mode in ("none", "inner", "input"):
        blk = ConvBlock(3, 16, ConvBlockConfig(n_convs=2, residual=mode, norm="batch", stride_1st=2)).eval()
        with torch.no_grad():
            assert blk(torch.randn(1, 3, 33, 35)).shape == (1, 16, 17, 18)
    # "input" adds projection params only when shapes differ; identity shortcut otherwise
    blk_proj = ConvBlock(3, 16, ConvBlockConfig(residual="input", stride_1st=2))
    assert any("skip" in k for k in blk_proj.state_dict())
    blk_id = ConvBlock(16, 16, ConvBlockConfig(residual="input"))
    assert not any("skip" in k for k in blk_id.state_dict())
    # full UNet with residual="input" incl. odd input size
    cfg = UNetConfig(in_channels=2, out_channels=1, intermediate_channels=[8, 16, 32],
                     block=ConvBlockConfig(n_convs=2, residual="input", norm="batch"))
    m = UNet(cfg).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 2, 61, 63)).shape == (1, 1, 61, 63)
    # checkpoint roundtrip preserves the mode
    p = tmp_path / "m.ckpt"
    save_checkpoint(m, p)
    assert load_unet_checkpoint(p).config.block.residual == "input"


def test_newer_schema_version_warns():
    d = UNetConfig().to_dict()
    d["schema_version"] = 999
    with pytest.warns(UserWarning, match="schema_version"):
        UNetConfig.from_dict(d)


def test_lightning_out_channels_alias():
    # legacy keyword still works
    net = LightningUNet(n_classes=2, intermediate_channels=[8, 16], input_channels=3)
    assert net.config.out_channels == 2
    assert net.hparams["n_classes"] == 2  # legacy hparam key preserved
    # new name works positionally and as keyword
    net2 = LightningUNet(2, [8, 16], input_channels=3)
    net3 = LightningUNet(out_channels=2, intermediate_channels=[8, 16], input_channels=3)
    assert net2.config == net3.config == net.config
    # conflicting values raise
    with pytest.raises(ValueError, match="Conflicting"):
        LightningUNet(out_channels=2, n_classes=3, intermediate_channels=[8, 16])
    # in_channels primary, input_channels legacy alias
    net4 = LightningUNet(out_channels=2, intermediate_channels=[8, 16], in_channels=3)
    assert net4.config.in_channels == 3
    assert net4.hparams["input_channels"] == 3  # legacy hparam key preserved
    net5 = LightningUNet(n_classes=2, intermediate_channels=[8, 16], input_channels=3)
    assert net5.config == net4.config
    with pytest.raises(ValueError, match="Conflicting"):
        LightningUNet(out_channels=2, intermediate_channels=[8, 16], in_channels=3, input_channels=5)
    # default in_channels=1 preserved
    assert LightningUNet(2, [8, 16]).config.in_channels == 1


def test_lightning_load_old_checkpoint_with_n_classes_hparam(tmp_path):
    # simulate an old checkpoint whose hyper_parameters only have n_classes
    net = LightningUNet(2, [8, 16, 32], input_channels=3, unet_kwargs={"n_convs_per_block": 2})
    path = tmp_path / "old.ckpt"
    import lightning
    old_hparams = {k: v for k, v in net.hparams.items() if k != "out_channels"}
    assert "n_classes" in old_hparams and "out_channels" not in old_hparams
    torch.save({"state_dict": net.state_dict(), "hyper_parameters": old_hparams,
                "pytorch-lightning_version": lightning.__version__}, path)
    loaded = LightningUNet.load_from_checkpoint(path)
    assert loaded.config.out_channels == 2
    assert loaded.config.block.n_convs == 2


def test_lightning_checkpoint_roundtrip_with_custom_config(tmp_path):
    # custom config differing from the legacy defaults
    cfg = UNetConfig(in_channels=3, out_channels=2, intermediate_channels=[8, 16, 32],
                     block=ConvBlockConfig(n_convs=2, norm="group", activation="gelu", residual=True))
    net = LightningUNet(2, [8, 16, 32], input_channels=3, config=cfg)
    path = tmp_path / "ln.ckpt"
    import lightning
    torch.save({"state_dict": net.state_dict(), "hyper_parameters": dict(net.hparams),
                "pytorch-lightning_version": lightning.__version__}, path)
    loaded = LightningUNet.load_from_checkpoint(path)
    assert loaded.config == cfg  # architecture must survive the round-trip, not fall back to defaults
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(net.eval()(x), loaded.eval()(x))
