"""Runtime RNG seeding."""

from __future__ import annotations

import random

import torch


def seed_training_rng(seed: int) -> None:
    """Seed Python and Torch RNG streams for a fresh runtime process."""
    assert seed >= 0
    random.seed(seed)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    torch.set_rng_state(cpu_generator.get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
