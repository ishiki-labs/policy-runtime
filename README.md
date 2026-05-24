# policy-runtime

Standalone runtime for traced robot policy bundles. Load a bundle, hand it pre-stacked observations, get a raw action chunk back. No dependency on the training repo.

## What's in a bundle

A directory produced by `model-playground/scripts/export_to_runtime_bundle.py`:

```
my_bundle/
├── manifest.json   # camera names + order, shapes, history strides, num_predictions
└── model.pt        # TorchScript-traced policy. Normalization + delta logic baked in.
```

The exported `model.pt` takes **raw** values in and returns **raw** values out:
- Input qpos: raw joint positions (e.g. radians) — the trace normalizes internally
- Input images: float32 `(B, 3, H, W)` in [0, 1]
- Output actions: raw joint targets (radians) — already denormalized, already delta-applied if model trained that way

That means the runtime never has to know whether the model was trained with delta or absolute actions, or what the normalization stats are.

## Install

```bash
uv add policy-runtime  # if published
# or, editable from a checkout:
uv pip install -e .
```

Deps: `torch`, `numpy`.

## Usage

```python
import numpy as np
from policy_runtime import Runtime

rt = Runtime.load("my_bundle/", device="cuda")
m = rt.manifest

# You maintain raw-qpos and raw-image ring buffers yourself.
# Then build the stacks per the manifest's shape spec:

qpos_stack    = np.zeros((1, m.num_qpos_tokens, m.robot_state_dim), np.float32)  # oldest first
current_qpos  = np.zeros((1, m.robot_state_dim), np.float32)
env_state     = np.zeros((1, m.env_state_dim), np.float32)
camera_images = {
    name: np.zeros((1, 3, m.image_chw[1], m.image_chw[2]), np.float32)  # float [0,1]
    for name in m.model_input_camera_names
}

action_chunk = rt.forward(
    qpos_stack=qpos_stack,
    current_qpos=current_qpos,
    env_state=env_state,
    camera_images=camera_images,
)
# action_chunk: (1, num_predictions, robot_state_dim) raw joint targets

# You pick which action to execute (chunk-and-pop, every-step, your own ensembling).
```

## What the caller is responsible for

| Concern | Why caller, not runtime |
|---|---|
| Image ring buffer (for `overhead_t-1` etc.) | Source-specific (cameras, threads, timestamps) |
| qpos ring buffer | Source-specific |
| Resize images to `manifest.image_chw[1:]` | Caller knows the source resolution |
| Convert uint8 → float [0,1] (or pass float directly) | Trivial; caller's choice of speed vs. dtype |
| Action selection from the chunk | Control-loop-specific (chunked, throttled, ensembled) |
| IK for ee_pose action spaces | Solver choice + URDF live with the robot |

For the action-selection piece, see `policy_runtime.helpers.ActionChunkQueue` — an optional helper for the common "chunk now, pop one per control step, refill on empty, optional temporal blend" pattern.

## Manifest schema

```json
{
  "schema_version": 1,
  "model_variant": "act_vae",
  "exported_from": "act_vae_scooping_halfchunk_qposdrop_v0 @ step 200000",
  "exported_at": "2026-05-24T18:00:00Z",

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
    "overhead_t-1", "overhead", "wrist_right_t-1", "wrist_right"
  ],

  "action_space": "joint",
  "device_hint": "cuda"
}
```

`model_input_camera_names` is the **positional order** the trace expects. The runtime packs `camera_images[name]` into this order before calling the traced graph.

## Supported variants (v1)

| Model variant | Supported | Notes |
|---|---|---|
| `act_vae` (joint actions, NONE env_state) | ✅ | Primary target |
| `act_vae` with `env_state` (e.g. SOURCE_AND_TARGET) | ❌ | env_state normalization not yet baked into trace |
| `act_vae` with reward head | ❌ | Trace currently drops reward output |
| `low_dim_act` | ✅ | No-camera variant |
| `detr_vae` | ⏳ | DETR's dynamic shapes may need `torch.jit.script` not trace |
| `simple_diffusion` | ❌ | Iterative loop needs unrolling |

| Action space | Supported | Notes |
|---|---|---|
| joint (absolute or delta) | ✅ | Delta is baked into the trace transparently |
| ee_pose / ee_pose_6d | ⏳ | Trace returns raw actions; caller does IK |

## Exporting a bundle

Run from the model-playground repo, in the training env:

```bash
conda activate aloha_trossen_plating
python scripts/export_to_runtime_bundle.py \
    --checkpoint_dir /mnt/nas/scratchpad/checkpoints/act_vae_scooping_halfchunk_qposdrop_v0 \
    --output_dir bundles/scooping_v0 \
    --validate
```

`--validate` runs both `InferencePolicy` (training-stack reference) and the exported bundle on identical observations and asserts actions agree to within 1e-4.
