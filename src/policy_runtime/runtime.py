from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from .bundle import Manifest, load_manifest

ArrayLike = Union[np.ndarray, torch.Tensor]


class Runtime:
    """Loads a policy bundle and runs the traced model.

    The traced model expects raw values and returns raw values — normalization
    and any delta-action math are baked into the graph at export time. The
    runtime's job is shape validation, dtype/device coercion, and packing the
    camera dict into the positional order the trace expects.
    """

    def __init__(self, model: torch.jit.ScriptModule, manifest: Manifest, device: torch.device) -> None:
        self._model = model
        self._manifest = manifest
        self._device = device

    @classmethod
    def load(cls, bundle_path: Union[str, Path], device: Union[str, torch.device] = "cuda") -> "Runtime":
        bundle_path = Path(bundle_path)
        if not bundle_path.is_dir():
            raise FileNotFoundError(f"Bundle path is not a directory: {bundle_path}")

        manifest_path = bundle_path / "manifest.json"
        model_path = bundle_path / "model.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {bundle_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model.pt in {bundle_path}")

        manifest = load_manifest(manifest_path)
        device = torch.device(device)
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()

        return cls(model=model, manifest=manifest, device=device)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(
        self,
        qpos_stack: ArrayLike,
        current_qpos: ArrayLike,
        env_state: ArrayLike,
        camera_images: dict[str, ArrayLike],
    ) -> np.ndarray:
        """Run one forward pass through the traced policy.

        Args:
            qpos_stack: (B, num_qpos_tokens, robot_state_dim) raw qpos history,
                oldest token first, last token = current qpos.
            current_qpos: (B, robot_state_dim) raw current qpos. The trace uses
                this for delta-action math regardless of whether the model was
                trained with delta or absolute actions.
            env_state: (B, env_state_dim) raw env state. Use a zero-size tensor
                of shape (B, 0) if env_state_dim == 0.
            camera_images: dict mapping every name in
                `manifest.model_input_camera_names` to a tensor of shape
                (B, 3, H, W), float32 in [0, 1], where (H, W) ==
                manifest.image_chw[1:].

        Returns:
            np.ndarray of shape (B, num_predictions, robot_state_dim) — raw
            joint targets, already denormalized and (if applicable) delta-applied.
        """
        m = self._manifest

        qpos_stack_t = self._to_tensor(qpos_stack, name="qpos_stack")
        current_qpos_t = self._to_tensor(current_qpos, name="current_qpos")
        env_state_t = self._to_tensor(env_state, name="env_state")

        self._check_shape(qpos_stack_t, ("B", m.num_qpos_tokens, m.robot_state_dim), "qpos_stack")
        batch = qpos_stack_t.shape[0]
        self._check_shape(current_qpos_t, (batch, m.robot_state_dim), "current_qpos")
        self._check_shape(env_state_t, (batch, m.env_state_dim), "env_state")

        missing = [n for n in m.model_input_camera_names if n not in camera_images]
        if missing:
            raise ValueError(
                f"camera_images missing required keys: {missing}. "
                f"Expected keys: {list(m.model_input_camera_names)}"
            )

        camera_tensors: list[torch.Tensor] = []
        for name in m.model_input_camera_names:
            t = self._to_tensor(camera_images[name], name=f"camera_images[{name!r}]")
            expected = (batch, m.image_chw[0], m.image_chw[1], m.image_chw[2])
            self._check_shape(t, expected, f"camera_images[{name!r}]")
            camera_tensors.append(t)

        with torch.inference_mode():
            out = self._model(qpos_stack_t, current_qpos_t, env_state_t, *camera_tensors)

        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"Traced model returned non-tensor of type {type(out).__name__}. "
                "Export wrapper should produce a single action-chunk tensor."
            )
        return out.detach().cpu().numpy()

    def _to_tensor(self, x: ArrayLike, *, name: str) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x).__name__}")
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        return t.to(self._device)

    @staticmethod
    def _check_shape(t: torch.Tensor, expected: tuple, name: str) -> None:
        actual = tuple(t.shape)
        if len(actual) != len(expected):
            raise ValueError(f"{name} has rank {len(actual)} ({actual}), expected rank {len(expected)} ({expected})")
        for i, (a, e) in enumerate(zip(actual, expected)):
            if isinstance(e, str):
                continue
            if a != e:
                raise ValueError(f"{name} has shape {actual}, expected {expected} (mismatch at dim {i})")
