"""Optional fixed state normalization for global-state learners.

Normalization is opt-in. The caller must supply explicit ``mean`` and ``std``
vectors matching ``state_dim``; no highway-specific shape is assumed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


class StateNormalizer(nn.Module):
    """Apply fixed mean/std normalization to a global state vector."""

    def __init__(
        self,
        state_dim: int,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)
        if mean_array.shape != (state_dim,) or std_array.shape != (state_dim,):
            raise ValueError(
                f"mean and std must have shape ({state_dim},); "
                f"received {mean_array.shape} and {std_array.shape}."
            )
        if np.any(std_array <= 0.0) or not np.all(np.isfinite(std_array)):
            raise ValueError("std values must be finite and positive.")
        if not np.all(np.isfinite(mean_array)):
            raise ValueError("mean values must be finite.")
        self.enabled = bool(enabled)
        self.register_buffer("mean", torch.as_tensor(mean_array))
        self.register_buffer("std", torch.as_tensor(std_array))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return states
        return (states - self.mean) / self.std


__all__ = ["StateNormalizer"]
