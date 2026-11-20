"""Candidate selection module (teammate-selection research).

Provides CARLA-, Gymnasium-, PyTorch- and V2X-channel-independent selectors
that choose ``k`` teammates from an eligible candidate pool, without modifying
the 56-dimensional observation contract.

Public API:
    Selection          -- immutable selection result
    NearestSelector    -- nearest-k selection (strategy "nearest_k")
    RandomSelector     -- random-k selection without replacement (strategy "random_k")
    LearnedSelector    -- placeholder for future learned selection (strategy "learned_k")
    make_selector      -- factory returning a selector for a configured mode
"""

from __future__ import annotations

from typing import Optional

from .base import Selection
from .learned_sel import LearnedSelector
from .random_sel import RandomSelector
from .radius import NearestSelector

_SELECTORS: dict[str, object] = {
    "nearest_k": NearestSelector,
    "random_k": RandomSelector,
    "learned_k": LearnedSelector,
}

# Modes that mean "no selection; keep the legacy full-pool behaviour."
_LEGACY_MODES = frozenset({None, "legacy", "disabled", "none", "off"})


def make_selector(
    mode: Optional[str],
) -> Optional[NearestSelector | RandomSelector | LearnedSelector]:
    """Return a selector instance for the configured selection ``mode``.

    Args:
        mode: One of ``"nearest_k"``, ``"random_k"``, ``"learned_k"``, or a
            legacy/disabled mode (``None``, ``"legacy"``, ``"disabled"``,
            ``"none"``, ``"off"``) which returns ``None`` meaning "no selection".

    Returns:
        A selector instance, or ``None`` for legacy/disabled modes.

    Raises:
        ValueError: if ``mode`` is not recognised. No silent fallback is
            performed.
    """
    if mode in _LEGACY_MODES:
        return None
    key = str(mode).strip().lower()
    if key in _LEGACY_MODES:
        return None
    if key not in _SELECTORS:
        raise ValueError(
            f"unknown selection mode {mode!r}; expected one of "
            "'nearest_k', 'random_k', 'learned_k', or a legacy/disabled mode"
        )
    return _SELECTORS[key]()


__all__ = [
    "LearnedSelector",
    "NearestSelector",
    "RandomSelector",
    "Selection",
    "make_selector",
]
