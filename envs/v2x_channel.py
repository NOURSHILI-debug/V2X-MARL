"""Pure-NumPy V2X broadcast channel (no CARLA / highway-env dependency).

Each sender broadcasts ``[x, y, vx, vy, heading, intent]``. The channel applies
a Euclidean range check, independent per-(sender, receiver, step) Bernoulli
packet loss and a fixed decision-step delay. It is deliberately agnostic about
who broadcasts: the caller decides whether background vehicles are included.

See ``docs/v2x_channel.md`` for the full semantics and open questions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from envs.obs_builder import (
    CANDIDATE_RANGE,
    CAND_FEATURES,
    M_CANDIDATES,
    V_MAX,
    to_ego_frame,
)

# Message indices: [x, y, vx, vy, heading, intent]
X, Y, VX, VY, HEADING, INTENT = range(6)
MESSAGE_FIELDS = 6


@dataclass(frozen=True)
class V2XConfig:
    """Immutable channel parameters with validation."""

    range_m: float = 50.0
    delay_steps: int = 0
    loss_prob: float = 0.0

    def __post_init__(self) -> None:
        try:
            range_m = float(self.range_m)
        except (TypeError, ValueError):
            raise ValueError("range_m must be a finite number >= 0") from None
        if not math.isfinite(range_m) or range_m < 0.0:
            raise ValueError("range_m must be a finite number >= 0")
        object.__setattr__(self, "range_m", range_m)

        delay = self.delay_steps
        if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)):
            raise ValueError("delay_steps must be a non-negative integer")
        delay = int(delay)
        if delay < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        object.__setattr__(self, "delay_steps", delay)

        try:
            loss_prob = float(self.loss_prob)
        except (TypeError, ValueError):
            raise ValueError("loss_prob must be a finite number in [0, 1]") from None
        if not math.isfinite(loss_prob) or not 0.0 <= loss_prob <= 1.0:
            raise ValueError("loss_prob must be a finite number in [0, 1]")
        object.__setattr__(self, "loss_prob", loss_prob)


class ReceivedMessage(NamedTuple):
    """One message delivered to a receiver."""

    sender_id: object
    age: int
    message: np.ndarray


def _as_message(message: object) -> np.ndarray:
    values = np.asarray(message, dtype=np.float64).reshape(-1)
    if values.shape != (MESSAGE_FIELDS,):
        raise ValueError(f"message must have {MESSAGE_FIELDS} values")
    if not np.all(np.isfinite(values)):
        raise ValueError("message values must be finite")
    return values


class V2XChannel:
    """Seeded V2X broadcast channel with range, loss and delay."""

    def __init__(self, config: Optional[V2XConfig] = None, seed: Optional[int] = None):
        if config is None:
            config = V2XConfig()
        if not isinstance(config, V2XConfig):
            raise TypeError("config must be a V2XConfig instance")
        self.config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._pending: Dict[int, List[Tuple[object, object, np.ndarray, int]]] = {}
        self._step_index = 0
        self._clear_counters()

    # ---- lifecycle ---------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Clear in-flight messages and counters, and replay a seeded stream.

        ``reset(None)`` replays the seed last used (initial or explicit).
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._pending.clear()
        self._step_index = 0
        self._clear_counters()

    def _clear_counters(self) -> None:
        self._sent = 0
        self._delivered = 0
        self._dropped_by_range = 0
        self._dropped_by_loss = 0
        self._age_sum = 0

    # ---- properties --------------------------------------------------------
    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def in_flight(self) -> int:
        return sum(len(batch) for batch in self._pending.values())

    # ---- simulation --------------------------------------------------------
    def step(
        self, senders: Sequence[Tuple[object, Sequence[float]]]
    ) -> Tuple[Dict[object, List[ReceivedMessage]], Dict[str, int]]:
        """Advance one decision step.

        ``senders`` is a list of ``(id, [x, y, vx, vy, heading, intent])``.
        Returns ``(received, stats)`` where ``received`` maps each receiver to
        its messages arriving this step and ``stats`` contains ``sent``,
        ``dropped_by_range``, ``dropped_by_loss``, ``delivered`` and the
        cumulative ``in_flight`` count.
        """
        parsed: List[Tuple[object, np.ndarray]] = []
        seen_ids = set()
        for entry in senders:
            try:
                sender_id, message = entry
            except (TypeError, ValueError):
                raise ValueError("each sender must be an (id, message) pair") from None
            if sender_id in seen_ids:
                raise ValueError("sender ids must be unique within a step")
            seen_ids.add(sender_id)
            parsed.append((sender_id, _as_message(message)))

        step = self._step_index
        stats = {
            "sent": 0,
            "dropped_by_range": 0,
            "dropped_by_loss": 0,
            "delivered": 0,
        }

        for sender_index, (sender_id, message) in enumerate(parsed):
            for receiver_index, (receiver_id, receiver_message) in enumerate(parsed):
                if sender_index == receiver_index or sender_id == receiver_id:
                    continue
                stats["sent"] += 1
                distance = math.hypot(
                    float(message[X] - receiver_message[X]),
                    float(message[Y] - receiver_message[Y]),
                )
                if distance > self.config.range_m:
                    stats["dropped_by_range"] += 1
                    continue
                if self.config.loss_prob > 0.0 and (
                    float(self._rng.random()) < self.config.loss_prob
                ):
                    stats["dropped_by_loss"] += 1
                    continue
                due = step + self.config.delay_steps
                self._pending.setdefault(due, []).append(
                    (sender_id, receiver_id, message.copy(), step)
                )

        received: Dict[object, List[ReceivedMessage]] = {}
        for sender_id, _ in parsed:
            received.setdefault(sender_id, [])
        for sender_id, receiver_id, message, sent_step in self._pending.pop(step, []):
            age = step - sent_step
            received.setdefault(receiver_id, []).append(
                ReceivedMessage(sender_id, age, message)
            )
            stats["delivered"] += 1
            self._age_sum += age

        self._sent += stats["sent"]
        self._delivered += stats["delivered"]
        self._dropped_by_range += stats["dropped_by_range"]
        self._dropped_by_loss += stats["dropped_by_loss"]
        self._step_index += 1
        stats["in_flight"] = self.in_flight
        return received, stats

    def episode_stats(self) -> Dict[str, Union[int, float]]:
        """Per-episode aggregated counters and mean delivery age (steps)."""
        return {
            "sent": self._sent,
            "delivered": self._delivered,
            "dropped_by_range": self._dropped_by_range,
            "dropped_by_loss": self._dropped_by_loss,
            "dropped": self._dropped_by_range + self._dropped_by_loss,
            "mean_age": self._age_sum / self._delivered if self._delivered else 0.0,
            "in_flight": self.in_flight,
        }


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _ego_state(ego: Union[Mapping[str, object], Sequence[float]]) -> Tuple[float, ...]:
    if isinstance(ego, Mapping):
        x = _finite(ego.get("x", 0.0))
        y = _finite(ego.get("y", 0.0))
        vx = _finite(ego.get("vx", 0.0))
        vy = _finite(ego.get("vy", 0.0))
        heading = _finite(ego.get("heading", ego.get("yaw", 0.0)))
        return x, y, vx, vy, heading
    values = np.asarray(ego, dtype=np.float64).reshape(-1)
    if values.size < 5:
        raise ValueError("ego sequence must contain at least [x, y, vx, vy, heading]")
    return tuple(float(value) for value in values[:5])


def _id_sort_key(sender_id: object) -> Tuple[int, Union[float, str]]:
    try:
        return 0, float(sender_id)
    except (TypeError, ValueError):
        return 1, str(sender_id)


def _clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def messages_to_candidates(
    received: Sequence[Union[ReceivedMessage, Sequence[object]]],
    ego: Union[Mapping[str, object], Sequence[float]],
    m: int = M_CANDIDATES,
) -> np.ndarray:
    """Convert one receiver's messages to the ``obs_builder`` candidate layout.

    Returns ``(m, 6)`` float32 rows of
    ``[rel_x/CANDIDATE_RANGE, rel_y/CANDIDATE_RANGE, rel_vx/V_MAX,
    rel_vy/V_MAX, wrapped_rel_heading/pi, valid]``, nearest first, zero padded.
    ``ego`` is a mapping with ``x, y, vx, vy, yaw`` (or ``heading``) or a
    sequence ``[x, y, vx, vy, heading]``.
    """
    ego_x, ego_y, ego_vx, ego_vy, ego_heading = _ego_state(ego)
    sortable: List[Tuple[float, Tuple[int, Union[float, str]], object, np.ndarray]] = []
    for item in received:
        sender_id, _age, message = item
        values = _as_message(message)
        distance = math.hypot(float(values[X] - ego_x), float(values[Y] - ego_y))
        sortable.append((distance, _id_sort_key(sender_id), sender_id, values))
    sortable.sort(key=lambda entry: (entry[0], entry[1]))

    block = np.zeros((m, CAND_FEATURES), dtype=np.float32)
    for index, (_, _, _, values) in enumerate(sortable[:m]):
        rel_x, rel_y = to_ego_frame(
            float(values[X]), float(values[Y]), ego_x, ego_y, ego_heading
        )
        rel_vx, rel_vy = to_ego_frame(
            float(values[VX] - ego_vx),
            float(values[VY] - ego_vy),
            0.0,
            0.0,
            ego_heading,
        )
        rel_heading = math.atan2(
            math.sin(float(values[HEADING] - ego_heading)),
            math.cos(float(values[HEADING] - ego_heading)),
        )
        block[index] = (
            _clip(rel_x / CANDIDATE_RANGE),
            _clip(rel_y / CANDIDATE_RANGE),
            _clip(rel_vx / V_MAX),
            _clip(rel_vy / V_MAX),
            _clip(rel_heading / math.pi),
            1.0,
        )
    return block


__all__ = [
    "HEADING",
    "INTENT",
    "MESSAGE_FIELDS",
    "ReceivedMessage",
    "V2XChannel",
    "V2XConfig",
    "VX",
    "VY",
    "X",
    "Y",
    "messages_to_candidates",
]
