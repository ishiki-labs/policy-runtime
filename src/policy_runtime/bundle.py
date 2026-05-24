from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Manifest:
    schema_version: int

    model_variant: str
    exported_from: str
    exported_at: str

    robot_state_dim: int
    env_state_dim: int
    num_predictions: int
    num_qpos_tokens: int

    image_chw: tuple[int, int, int]
    base_camera_names: tuple[str, ...]
    camera_history_length: int
    camera_history_stride: int

    qpos_history_length: int
    qpos_history_stride: int

    model_input_camera_names: tuple[str, ...]

    action_space: str
    device_hint: str

    extras: dict = field(default_factory=dict)


_SUPPORTED_SCHEMA_VERSIONS = {1}


def load_manifest(path: Path | str) -> Manifest:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    schema_version = raw["schema_version"]
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported manifest schema_version={schema_version}. "
            f"This policy-runtime supports versions {_SUPPORTED_SCHEMA_VERSIONS}."
        )

    known_keys = {f.name for f in Manifest.__dataclass_fields__.values()} - {"extras"}
    extras = {k: v for k, v in raw.items() if k not in known_keys}

    return Manifest(
        schema_version=schema_version,
        model_variant=raw["model_variant"],
        exported_from=raw["exported_from"],
        exported_at=raw["exported_at"],
        robot_state_dim=int(raw["robot_state_dim"]),
        env_state_dim=int(raw["env_state_dim"]),
        num_predictions=int(raw["num_predictions"]),
        num_qpos_tokens=int(raw["num_qpos_tokens"]),
        image_chw=tuple(raw["image_chw"]),
        base_camera_names=tuple(raw["base_camera_names"]),
        camera_history_length=int(raw["camera_history_length"]),
        camera_history_stride=int(raw["camera_history_stride"]),
        qpos_history_length=int(raw["qpos_history_length"]),
        qpos_history_stride=int(raw["qpos_history_stride"]),
        model_input_camera_names=tuple(raw["model_input_camera_names"]),
        action_space=raw["action_space"],
        device_hint=raw["device_hint"],
        extras=extras,
    )
