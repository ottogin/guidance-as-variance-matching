"""Calibration noise, seeded per (row, level) so a moments pass does not depend on batching or sharding."""
import torch

MAX_LEVELS = 1009      # row * 1009 + level is injective for level < 1009 (MDM's 1,000 DDPM levels fit)


def eps(seed, row, level, shape):
    """The standard-normal noise for calibration row `row` at sampler level `level`, drawn on the CPU.

    The same formula as the SD moments passes behind the paper, so
    a fresh pass sees the same noise as the stored one.
    """
    assert 0 <= level < MAX_LEVELS, level
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(row) * MAX_LEVELS + int(level))
    return torch.randn(tuple(shape), dtype=torch.float32, generator=g)


def eps_batch(seed, rows, level, shape, device="cpu"):
    """[len(rows), *shape] noise for several rows at one level."""
    return torch.stack([eps(seed, r, level, shape) for r in rows]).to(device)


def generator(seed, row, stream=0):
    """A CPU generator for any other per-row randomness (e.g. a conditioning-noise draw), keyed by `stream`."""
    assert 0 <= stream < 64, stream
    # offset by 2^62 so these seeds never coincide with eps()'s
    return torch.Generator().manual_seed(2 ** 62 + int(seed) * 1_000_003 + int(row) * 64 + int(stream))
