"""Learned candidate selection — future work.

This module establishes the common selector interface but does **not** implement
a learned algorithm. A future learned selector would score candidates (for
example with an attention network over the candidate features) and return the
top ``k``. Until then it raises :exc:`NotImplementedError` so the placeholder is
never mistaken for a working strategy.

The interface mirrors :class:`~grouping.radius.NearestSelector` and
:class:`~grouping.random_sel.RandomSelector`: ``select(pool, k=2, *, ego=None,
rng=None) -> Selection``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from .base import Selection


class LearnedSelector:
    """Placeholder for learned selection (not implemented in this milestone)."""

    name = "learned_k"

    def select(
        self,
        pool: Sequence[Mapping[str, Any]],
        k: int = 2,
        *,
        ego: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Selection:
        """Not implemented; raises :exc:`NotImplementedError`."""
        del pool, k, ego, rng
        raise NotImplementedError(
            "learned selection is not implemented in Milestone 2; "
            "use NearestSelector or RandomSelector instead"
        )
