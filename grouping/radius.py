"""Nearest-k candidate selection (baseline: the k closest candidates).

Strategy name: ``nearest_k`` (not "radius"): this selects the ``k`` nearest
eligible candidates rather than applying a distance threshold.

CARLA-independent: operates on plain world-frame numeric records only. Distance
is Euclidean in the world (x, y) plane relative to an ego reference point.
Tie-breaking is deterministic: distance ascending, then numeric actor id
ascending.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Optional

import numpy as np

from .base import Selection, _id_sort_key, _validate_pool


class NearestSelector:
    """Select the k closest candidates to an ego reference point.

    The selector never mutates the input pool or its records. Selected records
    are shared references to the input records.
    """

    name = "nearest_k"

    def select(
        self,
        pool: Sequence[Mapping[str, Any]],
        k: int = 2,
        *,
        ego: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Selection:
        """Select up to ``k`` nearest candidates.

        Args:
            pool: Eligible candidate records, each a mapping with ``actor_id``,
                ``x`` and ``y`` (world coordinates). Extra keys are preserved.
            k: Requested number of teammates (>= 1).
            ego: Reference point with ``x``/``y`` world coordinates. Defaults to
                the origin when omitted.
            rng: Accepted for interface uniformity; nearest-k is deterministic
                and does not consume randomness.

        Returns:
            A :class:`Selection` with real records first (nearest first) followed
            by padding slots when ``len(pool) < k``. ``pool_size`` is the number
            of validated candidates; ``missing_slots`` is ``k - selected_count``.
        """
        del rng  # nearest-k is deterministic; rng is part of the common interface

        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")

        ego_x = 0.0
        ego_y = 0.0
        if ego is not None:
            ego_x = float(ego.get("x", 0.0))
            ego_y = float(ego.get("y", 0.0))
            if not (math.isfinite(ego_x) and math.isfinite(ego_y)):
                raise ValueError("ego x/y must be finite")

        validated = _validate_pool(pool)
        scored: list[tuple[float, tuple[int, Any], int, Any, Mapping[str, Any]]] = []
        for index, actor_id, x, y, candidate in validated:
            distance = math.hypot(x - ego_x, y - ego_y)
            scored.append((distance, _id_sort_key(actor_id), index, actor_id, candidate))

        scored.sort(key=lambda item: (item[0], item[1]))

        take = min(k, len(scored))
        chosen = scored[:take]
        indices = tuple(item[2] for item in chosen)
        ids = tuple(item[3] for item in chosen)
        records = tuple(item[4] for item in chosen)
        scores = tuple(item[0] for item in chosen)

        pad = k - take
        indices = indices + (-1,) * pad
        ids = ids + (None,) * pad
        records = records + (None,) * pad
        scores = scores + (None,) * pad

        return Selection(
            strategy=self.name,
            k=k,
            indices=indices,
            ids=ids,
            records=records,
            scores=scores,
            pool_size=len(validated),
        )
