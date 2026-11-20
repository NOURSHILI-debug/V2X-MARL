from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.milestone9.logging import (
    ExperimentLayout,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    derive_ending_reason,
    sha256_file,
)


def test_atomic_write_and_checksums(tmp_path: Path):
    p = tmp_path / "data.json"
    atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(p.read_text(encoding="utf-8"))["a"] == 1
    h1 = sha256_file(p)
    atomic_write_text(tmp_path / "note.txt", "hello")
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"
    csv_path = tmp_path / "rows.csv"
    atomic_write_csv(csv_path, [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}], fieldnames=["x", "y"])
    text = csv_path.read_text(encoding="utf-8")
    assert "x,y" in text.replace("\r\n", "\n")
    assert sha256_file(p) == h1


def test_experiment_layout_manifest_and_no_overwrite(tmp_path: Path):
    layout = ExperimentLayout(tmp_path, "exp_test")
    assert (layout.base / "episodes").is_dir()
    assert (layout.base / "metadata" / "manifest.json").exists()
    f = layout.base / "episodes" / "e.json"
    atomic_write_json(f, {"ok": True})
    rec = layout.register_file(f, "episode", extra={"episode_id": "e1"})
    assert rec["sha256"] == sha256_file(f)
    layout.register_episode({"episode_id": "e1", "execution_status": "success"})
    layout.update_status("completed")
    manifest_path = layout.save_manifest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "exp_test"
    assert manifest["status"] == "completed"
    assert manifest["files"][0]["path"].replace("\\", "/").endswith("episodes/e.json")
    assert manifest["episodes"][0]["episode_id"] == "e1"

    with pytest.raises(FileExistsError):
        ExperimentLayout(tmp_path, "exp_test")


def test_derive_ending_reason_from_info():
    assert derive_ending_reason({"success": True, "goal_reached": True, "steps": 10}, 300) == "goal"
    assert derive_ending_reason({"collision": True, "steps": 10}, 300) == "collision"
    assert derive_ending_reason({"off_road": True, "steps": 10}, 300) == "off_road"
    assert derive_ending_reason({"stuck": True, "steps": 10}, 300) == "stuck"
    assert derive_ending_reason({"steps": 300, "success": False}, 300) == "timeout"
    assert derive_ending_reason({"ending_reason": "timeout"}, 300) == "timeout"
    assert derive_ending_reason({}, 300) is None
