"""Standardized logging and artifact management for Milestone 9.

Additive, CARLA-free utilities:
- experiment directory layout under results/milestone9/
- atomic JSON/CSV/text writing
- checksums and manifest generation
- episode/summary schema helpers

No existing experiment directories/results are overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "milestone9"

SUBDIRS = ("raw", "episodes", "summaries", "statistics", "figures", "tables", "metadata")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha_short(cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd or REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def config_hash(config_dict: Mapping[str, Any], length: int = 8) -> str:
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def make_experiment_id(config_dict: Mapping[str, Any], label: str = "m9") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{label}_{ts}_{git_sha_short()}_{config_hash(config_dict)}"


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    text = json.dumps(data, indent=indent, sort_keys=False, default=json_default) + "\n"
    atomic_write_text(path, text)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class ExperimentLayout:
    """Manages a single experiment output directory without overwriting existing runs."""

    def __init__(self, root: Path, experiment_id: str, *, allow_existing_empty: bool = False):
        self.root = Path(root)
        self.experiment_id = str(experiment_id)
        self.base = self.root / self.experiment_id
        self.paths = {name: self.base / name for name in SUBDIRS}
        self.manifest_path = self.paths["metadata"] / "manifest.json"
        self._manifest: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "created_utc": utc_now_iso(),
            "git_sha": git_sha_short(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "files": [],
            "episodes": [],
            "status": "running",
        }
        self._prepare(allow_existing_empty=allow_existing_empty)

    @property
    def manifest(self) -> Dict[str, Any]:
        return self._manifest

    def _prepare(self, allow_existing_empty: bool = False) -> None:
        if self.base.exists():
            has_content = any(self.base.iterdir())
            if has_content or self.manifest_path.exists():
                raise FileExistsError(
                    f"Experiment directory already exists and is not empty: {self.base}. "
                    "Provide a new experiment ID or explicit safe output path."
                )
            if not allow_existing_empty and has_content:
                raise FileExistsError(f"Experiment directory exists: {self.base}")
        for p in self.paths.values():
            ensure_dir(p)
        # write initial manifest atomically
        atomic_write_json(self.manifest_path, self._manifest)

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.base.resolve()))
        except Exception:
            return str(path)

    def register_file(self, path: Path, kind: str, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        path = Path(path)
        rec = {
            "path": self.rel(path),
            "kind": kind,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "registered_utc": utc_now_iso(),
        }
        if extra:
            rec.update(dict(extra))
        self._manifest.setdefault("files", []).append(rec)
        return rec

    def register_episode(self, record: Mapping[str, Any]) -> None:
        self._manifest.setdefault("episodes", []).append(dict(record))

    def update_status(self, status: str, **extra: Any) -> None:
        allowed = {"running", "incomplete", "failed", "completed"}
        if status not in allowed:
            raise ValueError(f"invalid manifest status {status!r}; expected one of {sorted(allowed)}")
        self._manifest["status"] = status
        self._manifest["updated_utc"] = utc_now_iso()
        for k, v in extra.items():
            self._manifest[k] = v

    def save_manifest(self) -> Path:
        self._manifest["updated_utc"] = utc_now_iso()
        atomic_write_json(self.manifest_path, self._manifest)
        return self.manifest_path


def derive_ending_reason(info: Mapping[str, Any], env_max_steps: Optional[int] = None) -> Optional[str]:
    """Derive canonical ending reason from env info/flags.

    envs.termination.build_step_info sets terminal flags but does not store an
    'ending_reason' string key; this helper provides a consistent derived value.
    """
    if info is None:
        return None
    if "ending_reason" in info and info.get("ending_reason") is not None:
        return str(info.get("ending_reason"))
    if not info:
        return None
    if info.get("success") or info.get("goal_reached"):
        # team goal success
        return "goal"
    if info.get("collision"):
        return "collision"
    if info.get("off_road"):
        return "off_road"
    if info.get("stuck"):
        return "stuck"
    steps = info.get("steps")
    if env_max_steps is not None and steps is not None:
        try:
            if int(steps) >= int(env_max_steps) and not info.get("success"):
                return "timeout"
        except Exception:
            pass
    # if episode ended by truncation without explicit env reason, caller should pass context;
    # return None rather than guessing incorrectly.
    return None


def normalize_ids(ids: Any) -> List[Any]:
    if ids is None:
        return []
    out = []
    for x in ids:
        if x is None:
            out.append(None)
        else:
            out.append(x)
    return out


def selection_summary_from_info(per_agent_info: Mapping[str, Any]) -> Dict[str, Any]:
    sel = (per_agent_info or {}).get("selection") or {}
    return {
        "selected_ids": normalize_ids(sel.get("selected_ids")),
        "pool_size": sel.get("pool_size"),
        "selected_count": sel.get("selected_count"),
        "missing_slots": sel.get("missing_slots"),
        "strategy": sel.get("strategy"),
        "scores": normalize_ids(sel.get("scores")) if "scores" in sel else None,
    }


def v2x_received_from_info(per_agent_info: Mapping[str, Any]) -> Dict[str, Any]:
    recv = (per_agent_info or {}).get("v2x_received") or {}
    return {
        "count": recv.get("count"),
        "sender_ids": normalize_ids(recv.get("sender_ids")),
        "ages": normalize_ids(recv.get("ages")),
    }


def scenario_signature_from_reset(reset_info: Mapping[str, Any], agents: Sequence[str]) -> Dict[str, Any]:
    """Extract comparable initial-scenario metadata from env reset info/observations context.

    Callers should augment with per-agent initial transforms/routes/speeds collected
    from env state after reset when available.
    """
    per_agent = (reset_info or {}).get("per_agent", {}) or {}
    return {
        "seed": reset_info.get("seed"),
        "town": reset_info.get("town"),
        "spawn_mode": reset_info.get("spawn_mode"),
        "spawn_cluster_max_distance": reset_info.get("spawn_cluster_max_distance"),
        "route_applied": reset_info.get("route_applied"),
        "per_agent_reset": {aid: per_agent.get(aid, {}) for aid in agents},
    }


__all__ = [
    "DEFAULT_RESULTS_ROOT",
    "ExperimentLayout",
    "SUBDIRS",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_text",
    "config_hash",
    "derive_ending_reason",
    "ensure_dir",
    "git_sha_short",
    "json_default",
    "make_experiment_id",
    "normalize_ids",
    "scenario_signature_from_reset",
    "selection_summary_from_info",
    "sha256_file",
    "utc_now_iso",
    "v2x_received_from_info",
]
