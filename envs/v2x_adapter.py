"""Adapter converting V2X received messages into candidate records.

Translates :class:`envs.v2x_channel.ReceivedMessage` objects into the
world-frame candidate mapping expected by :func:`envs.obs_builder.build_observation`.

This module is the *only* component responsible for translating V2X messages
into the candidate representation. It is intentionally free of CARLA
dependencies.

The adapter does NOT:

* filter by communication range (the channel already does this);
* filter by the 50 m candidate observation radius (build_observation does this);
* normalize features (build_observation does this exactly once);
* modify Nour's V2X channel implementation.

Message ``age`` and ``intent`` are preserved as metadata fields on each record.
They are ignored by ``build_observation``'s six-feature candidate block and are
kept available for logging / future use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from envs.v2x_channel import (
    HEADING,
    INTENT,
    ReceivedMessage,
    VX,
    VY,
    X,
    Y,
)


def received_to_candidates(
    received: Sequence[ReceivedMessage],
    ego: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Convert one receiver's V2X messages to world-frame candidate records.

    Each ``ReceivedMessage`` carries a six-field message
    ``[x, y, vx, vy, heading, intent]`` in world coordinates captured at
    transmission time. This adapter converts them into the dict layout expected
    by :func:`envs.obs_builder.build_observation`:

    ``{"actor_id", "x", "y", "vx", "vy", "yaw", "valid"}``

    The raw world coordinates are used (no normalization). The observation
    builder applies its own 50 m candidate-range filter, sorting, and
    normalization, so this adapter deliberately does none of those.

    Args:
        received: Received messages for one ego, as returned by
            ``V2XChannel.step()`` (``received[receiver_id]``).
        ego: Ego state mapping with ``x``/``y`` (world coordinates). Used only
            for documentation / future eligibility; currently unused because
            eligibility filtering is the observation builder's responsibility.

    Returns:
        A list of world-frame candidate dicts, one per received message, in the
        order they appear in ``received``. Each dict includes ``age`` and
        ``intent`` metadata fields in addition to the observation fields.
    """
    del ego  # eligibility / range filtering is build_observation's responsibility

    candidates: list[dict[str, Any]] = []
    for message in received:
        values = message.message
        candidates.append(
            {
                "actor_id": message.sender_id,
                "x": float(values[X]),
                "y": float(values[Y]),
                "vx": float(values[VX]),
                "vy": float(values[VY]),
                "yaw": float(values[HEADING]),
                "valid": True,
                # Metadata (not part of build_observation's 6-feature block):
                "age": message.age,
                "intent": float(values[INTENT]),
            }
        )
    return candidates


__all__ = ["received_to_candidates"]
