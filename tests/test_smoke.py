"""Smoke tests: imports + manifest round-trip. No actual model required."""
from __future__ import annotations

import json
from pathlib import Path


def test_imports():
    from policy_runtime import Manifest, Runtime, load_manifest  # noqa: F401


def test_manifest_round_trip(tmp_path: Path):
    from policy_runtime import load_manifest

    manifest_dict = {
        "schema_version": 1,
        "model_variant": "act_vae",
        "exported_from": "test_checkpoint @ step 100",
        "exported_at": "2026-05-24T00:00:00Z",
        "robot_state_dim": 7,
        "env_state_dim": 0,
        "num_predictions": 50,
        "num_qpos_tokens": 50,
        "image_chw": [3, 480, 848],
        "base_camera_names": ["overhead", "wrist_right"],
        "camera_history_length": 1,
        "camera_history_stride": 50,
        "qpos_history_length": 49,
        "qpos_history_stride": 1,
        "model_input_camera_names": [
            "overhead_t-1", "overhead", "wrist_right_t-1", "wrist_right",
        ],
        "action_space": "joint",
        "device_hint": "cuda",
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest_dict))
    m = load_manifest(p)
    assert m.robot_state_dim == 7
    assert m.num_qpos_tokens == 50
    assert m.model_input_camera_names == (
        "overhead_t-1", "overhead", "wrist_right_t-1", "wrist_right",
    )
    assert m.image_chw == (3, 480, 848)


def test_manifest_rejects_unknown_schema(tmp_path: Path):
    import pytest

    from policy_runtime import load_manifest

    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(p)
