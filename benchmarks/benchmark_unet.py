"""Benchmark: legacy vs new UNet, and eager vs compiled vs channels-last (MPS + CPU)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests')))


from calmutils.nn.unet import UNet, UNetConfig, ConvBlockConfig
from unet_legacy_snapshot import UNet as LegacyUNet

# config close to nucleolin_unet_config.json (plane_sliding_window=5, [64,128,128])
INTERMEDIATES = [64, 128, 128]
IN_CH, OUT_CH, BATCH, SIZE = 5, 1, 4, 128


def bench(model, x, n_warm=5, n_iters=30):
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        if x.device.type == 'mps':
            torch.mps.synchronize()
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(x)
            if x.device.type == 'mps':
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000  # median ms


def run(device):
    print(f"\n=== device: {device} | batch={BATCH} {SIZE}x{SIZE} in_ch={IN_CH} intermediates={INTERMEDIATES} ===")
    x = torch.randn(BATCH, IN_CH, SIZE, SIZE, device=device)

    torch.manual_seed(0)
    legacy = LegacyUNet(IN_CH, INTERMEDIATES, OUT_CH, dimensionality=2,
                        residual_conv_blocks=True, n_convs_per_block=3, batch_norm=True).to(device).eval()
    torch.manual_seed(0)
    new = UNet(IN_CH, INTERMEDIATES, OUT_CH, dimensionality=2,
               residual_conv_blocks=True, n_convs_per_block=3, batch_norm=True).to(device).eval()

    # verify parity on this device
    with torch.no_grad():
        from calmutils.nn.unet import load_state_dict_flexible
        load_state_dict_flexible(new, legacy.state_dict())
        d = (legacy(x) - new(x)).abs().max().item()
    print(f"parity max abs diff: {d:.2e}")

    print(f"legacy (snapshot) : {bench(legacy, x):8.2f} ms")
    print(f"new eager         : {bench(new, x):8.2f} ms")

    from calmutils.nn.unet import remap_legacy_state_dict
    new_cl = UNet(UNetConfig(in_channels=IN_CH, out_channels=OUT_CH, intermediate_channels=INTERMEDIATES,
                             block=ConvBlockConfig(n_convs=3, residual=True, norm='batch')))
    new_cl.load_state_dict(remap_legacy_state_dict(legacy.state_dict()))
    new_cl = new_cl.to(device).eval().to(memory_format=torch.channels_last)
    x_cl = x.to(memory_format=torch.channels_last)
    print(f"new channels_last : {bench(new_cl, x_cl):8.2f} ms")

    try:
        compiled = torch.compile(new, mode='max-autotune')
        print(f"new torch.compile : {bench(compiled, x, n_warm=10):8.2f} ms")
    except Exception as e:
        print(f"new torch.compile : failed ({e})")

    # transpose upsampling variant
    cfg_t = UNetConfig(in_channels=IN_CH, out_channels=OUT_CH, intermediate_channels=INTERMEDIATES,
                       block=ConvBlockConfig(n_convs=3, residual=True, norm='batch'), upsampling='transpose')
    m_t = UNet(cfg_t).to(device).eval()
    print(f"new transpose-up  : {bench(m_t, x):8.2f} ms")


if __name__ == '__main__':
    run(torch.device('mps'))
    run(torch.device('cpu'))
