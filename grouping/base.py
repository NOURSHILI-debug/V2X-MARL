"""Shared selection result representation and validation helpers.

This module is intentionally free of CARLA, Gymnasium, PyTorch and V2X-channel
dependencies. Selectors operate on plain world-frame candidate dictionaries and
return an immutable :class:`Selection` result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class Selection:
    """Immutable result of selecting up to ``k`` candidates from a pool.

    Attributes:
        strategy: Selector name (for example ``"nearest_k"`` or ``"random_k"``).
        k: Requested number of teammates.
        indices: Position of each selected record in the *original input pool*,
            or ``-1`` for a padding slot.
        ids: Actor id of each selected candidate, or ``None`` for padding.
        records: The selected candidate records (shared references, not copies),
            or ``None`` for padding. Use :meth:`selected_records` for the
            compact list suitable for :func:`envs.obs_builder.build_observation`.
        scores: Distance (nearest-k) or score when available, or ``None`` for
            padding / strategies without a meaningful score.
        pool_size: Number of eligible candidates in the input pool.
    """

    strategy: str
    k: int
    indices: tuple[int, ...]
    ids: tuple[Any | None, ...]
    records: tuple[Mapping[str, Any] | None, ...]
    scores: tuple[float | None, ...]
    pool_size: int

    @property
    def selected_count(self) -> int:
        """Number of real (non-padded) selections; <= k and <= pool_size."""
        return sum(1 for record in self.records if record is not None)

    @property
    def missing_slots(self) -> int:
        """Padding count ``k - selected_count``; never negative."""
        return max(0, self.k - self.selected_count)

    def selected_records(self) -> list[Mapping[str, Any]]:
        """Non-padded records in selection order, for the observation builder."""
        return [record for record in self.records if record is not None]


def _validate_pool(
    pool: Sequence[Mapping[str, Any]],
) -> list[tuple[int, Any, float, float, Mapping[str, Any]]]:
    """Validate every candidate record and return ``(index, id, x, y, record)``.

    Raises:
        ValueError: if a record is not a mapping, lacks ``actor_id``/``x``/``y``,
            has non-finite ``x``/``y``, or contains a duplicate ``actor_id``.
    """
    validated: list[tuple[int, Any, float, float, Mapping[str, Any]]] = []
    seen_ids: set[Any] = set()
    for index, candidate in enumerate(pool):
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"candidate at index {index} is not a mapping: {candidate!r}"
            )
        actor_id = candidate.get("actor_id")
        if actor_id is None:
            raise ValueError(f"candidate at index {index} is missing 'actor_id'")
        if actor_id in seen_ids:
            raise ValueError(f"duplicate actor_id {actor_id!r} at index {index}")
        seen_ids.add(actor_id)
        x = candidate.get("x")
        y = candidate.get("y")
        if x is None or y is None:
            raise ValueError(
                f"candidate {actor_id!r} at index {index} is missing 'x' or 'y'"
            )
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate {actor_id!r} at index {index} has non-numeric x/y"
            ) from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(
                f"candidate {actor_id!r} at index {index} has non-finite x/y"
            )
        validated.append((index, actor_id, x, y, candidate))
    return validated


def _id_sort_key(actor_id: Any) -> tuple[int, Any]:
    """Deterministic tie-break key: numeric ids first (by value), then by str."""
    if isinstance(actor_id, bool):
        return (1, str(actor_id))
    if isinstance(actor_id, (int,)) and not isinstance(actor_id, bool):
        return (0, int(actor_id))
    if isinstance(actor_id, float) and math.isfinite(actor_id) and actor_id.is_integer():
        return (0, int(actor_id))
    return (1, str(actor_id))
