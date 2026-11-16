"""Candidate selection module (teammate-selection research).

Provides CARLA-, Gymnasium-, PyTorch- and V2X-channel-independent selectors
that choose ``k`` teammates from an eligible candidate pool, without modifying
the 56-dimensional observation contract.

Public API:
    Selection          -- immutable selection result
    NearestSelector    -- nearest-k selection (strategy "nearest_k")
    RandomSelector     -- random-k selection without replacement (strategy "random_k")
    LearnedSelector    -- placeholder for future learned selection (strategy "learned_k")
"""

from __future__ import annotations

from .base import Selection
from .learned_sel import LearnedSelector
from .random_sel import RandomSelector
from .radius import NearestSelector

__all__ = [
    "LearnedSelector",
    "NearestSelector",
    "RandomSelector",
    "Selection",
]
