"""Random-k candidate selection (baseline: k distinct candidates without replacement).

Strategy name: ``random_k``.

CARLA-independent: operates on plain world-frame numeric records only. Randomness
is controlled exclusively by an explicit :class:`numpy.random.Generator`; the
global NumPy RNG is never used. Candidates are sampled without replacement, so
selected ids are always unique.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from .base import Selection, _validate_pool


class RandomSelector:
    """Select k distinct candidates uniformly without replacement.

    The selector never mutates the input pool or its records. Selected records
    are shared references to the input records.
    """

    name = "random_k"

    def select(
        self,
        pool: Sequence[Mapping[str, Any]],
        k: int = 2,
        *,
        ego: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Selection:
        """Select up to ``k`` distinct candidates without replacement.

        Args:
            pool: Eligible candidate records, each a mapping with ``actor_id``,
                ``x`` and ``y`` (world coordinates). Extra keys are preserved.
            k: Requested number of teammates (>= 1).
            ego: Accepted for interface uniformity; random-k does not use it.
            rng: Required :class:`numpy.random.Generator`. Reproducible when the
                same generator (or same seed) is supplied.

        Returns:
            A :class:`Selection` with real records first followed by padding
            slots when ``len(pool) < k``. ``pool_size`` is the number of
            validated candidates; ``missing_slots`` is ``k - selected_count``.

        Raises:
            ValueError: if ``rng`` is not provided or ``k < 1``.
        """
        del ego  # random-k does not use a reference point

        if rng is None:
            raise ValueError("random selection requires an explicit numpy.random.Generator")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")

        validated = _validate_pool(pool)
        n = len(validated)
        take = min(k, n)

        if take == 0:
            indices: tuple[int, ...] = ()
            ids: tuple[Any, ...] = ()
            records: tuple[Mapping[str, Any] | None, ...] = ()
            scores: tuple[float | None, ...] = ()
        else:
            # Permute pool positions and take the first `take` -> without replacement.
            chosen_positions = tuple(int(p) for p in rng.permutation(n)[:take])
            # Return chosen candidates in ascending input-pool order for stable metadata.
            chosen_positions = tuple(sorted(chosen_positions))
            chosen = [validated[pos] for pos in chosen_positions]
            indices = tuple(item[0] for item in chosen)
            ids = tuple(item[1] for item in chosen)
            records = tuple(item[4] for item in chosen)
            scores = (None,) * take  # random-k has no distance/score

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
            pool_size=n,
        )
