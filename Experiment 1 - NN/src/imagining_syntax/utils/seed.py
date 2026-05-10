"""
Centralized seeding for reproducible runs.

`set_global_seed(seed)` seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA),
and configures cuDNN for determinism. It does NOT enable
`torch.use_deterministic_algorithms(True)`, because some PyTorch ops (e.g. some
attention kernels) lack deterministic implementations and would error or run
much slower. For variance studies we want real seed-induced variance, not
bitwise-identical runs; cuDNN determinism + manual_seed is sufficient to keep
seed-to-seed variance to ≤O(1e-3) on small models like ours.

`make_torch_generator(seed)` returns a `torch.Generator` for use with
`DataLoader(generator=...)` so that batch shuffling is reproducible.
"""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - cuda branch unreachable when CUDA_VISIBLE_DEVICES="" (force_cpu fixture)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
