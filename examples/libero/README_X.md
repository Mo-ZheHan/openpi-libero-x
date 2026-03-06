# LIBERO-X Evaluation for pi0.5

This guide describes how to evaluate the `pi05_libero` model on LIBERO with [LIBERO-X](https://github.com/ljn24/LIBERO-X) perturbations (lighting, texture, view, robot state, layout, object swap/scale, etc.).

## Prerequisites

- `uv` package manager installed
- LIBERO-X git submodule initialized:
  ```bash
  git submodule update --init --recursive
  ```

## Environment Setup

The evaluation uses a **client-server architecture**. The model server runs in the main openpi environment, while the evaluation client runs in a separate Python 3.11 environment (required by LIBERO-X).

### 1. Create Python 3.11 virtual environment

```bash
uv venv --python 3.11 examples/libero/.venv_x
source examples/libero/.venv_x/bin/activate
```

### 2. Install LIBERO-X (includes patched LIBERO + libero_x)

```bash
uv pip install -e third_party/LIBERO-X
```

> This will automatically apply necessary patches to the LIBERO source code via the hatch build hook (custom_asset_dir, `__init__.py`, smooth_light_gray_plaster texture).

### 3. Install openpi client

```bash
uv pip install -e packages/openpi-client
```

### 4. Install additional dependencies

```bash
uv pip install imageio imageio-ffmpeg tyro tqdm
```

## Running Evaluation

### Terminal 1 — Model Server (main openpi env)

If building `av` from source, you need FFmpeg 7 dev libraries (e.g., via conda):

```bash
conda install -c conda-forge ffmpeg=7 pkg-config -y
```

Then start the server with `PKG_CONFIG_PATH` and `CUDA_VISIBLE_DEVICES` set:

> **Note:** JAX replicates the model on **all visible GPUs** by default (wasting memory).
> Use `CUDA_VISIBLE_DEVICES` to restrict to a single GPU. pi0.5-libero fits on one 4090 (24GB).

```bash
# Serve the default pi0.5 LIBERO checkpoint (downloads from GCS)
CUDA_VISIBLE_DEVICES=0 \
PKG_CONFIG_PATH=/home/mzh/miniconda3/lib/pkgconfig:$PKG_CONFIG_PATH \
LD_LIBRARY_PATH=/home/mzh/miniconda3/lib:$LD_LIBRARY_PATH \
uv run scripts/serve_policy.py --env LIBERO

# Or serve a local checkpoint:
CUDA_VISIBLE_DEVICES=0 \
PKG_CONFIG_PATH=/home/mzh/miniconda3/lib/pkgconfig:$PKG_CONFIG_PATH \
LD_LIBRARY_PATH=/home/mzh/miniconda3/lib:$LD_LIBRARY_PATH \
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /path/to/pi05_libero
```

### Terminal 2 — Evaluation Client (LIBERO-X env)

```bash
source examples/libero/.venv_x/bin/activate

# Run with perturbations (e.g., full_perturbation.yaml)
python examples/libero/main_x.py \
    --libero-x third_party/LIBERO-X/perturbations/full_perturbation.yaml

# Run without perturbations (equivalent to the original main.py)
python examples/libero/main_x.py

# Customize task suite and other options
python examples/libero/main_x.py \
    --libero-x third_party/LIBERO-X/perturbations/dim_lighting.yaml \
    --task-suite-name libero_10 \
    --num-trials-per-task 20
```

### Running Multiple Evaluations in Parallel

You can run multiple evaluations simultaneously on different GPUs. Each needs its own server (different GPU + port) and client (matching port + separate video dir).

**Server on GPU 1 (port 8001):**

```bash
CUDA_VISIBLE_DEVICES=1 \
PKG_CONFIG_PATH=/home/mzh/miniconda3/lib/pkgconfig:$PKG_CONFIG_PATH \
LD_LIBRARY_PATH=/home/mzh/miniconda3/lib:$LD_LIBRARY_PATH \
uv run scripts/serve_policy.py --env LIBERO --port 8001 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /path/to/pi05_libero
```

**Client connecting to port 8001:**

```bash
source examples/libero/.venv_x/bin/activate
python examples/libero/main_x.py \
    --libero-x third_party/LIBERO-X/perturbations/milk_perturbation.yaml \
    --task-suite-name libero_spatial \
    --num-trials-per-task 50 \
    --port 8001 \
    --video-out-path data/libero/videos_milk
```

> **Important:** `--port` must come **before** `policy:checkpoint` in the server command (tyro subcommand ordering).
> Use `--video-out-path` to avoid overwriting videos from other parallel runs.

## Available Perturbation Configs

Pre-defined configs are in `third_party/LIBERO-X/perturbations/`:

| Config | Description |
|--------|-------------|
| `dim_lighting.yaml` | Dim lighting only |
| `side_view.yaml` | Side camera view only |
| `visual_only.yaml` | Dim lighting + side view |
| `full_perturbation.yaml` | Lighting + texture + recolor + object scale |

You can also create custom YAML configs. Available perturbation dimensions:

```yaml
lighting: bright               # dim | bright | cool
texture: checkerboard          # warm | modern | checkerboard
view: side                     # side | top
robot_state: [0.0, 0.0, 0.05] # EE displacement [x, y, z] in metres
recolor_robot: true            # true | false
layout: distractor             # perturb | distractor (libero_10 only)
object_swap:                   # old_type: new_type
  akita_black_bowl: milk
language_swap:                 # old_text: new_text
  "black bowl": "milk carton"
object_scale:
  akita_black_bowl: 0.65
```

## Troubleshooting

- **EGL errors**: Try setting `MUJOCO_GL=osmesa` or `MUJOCO_GL=glx` before running the client.
- **Patches not applied**: Run `uv pip install -e third_party/LIBERO-X --force-reinstall` to re-trigger the build hook.
- **numpy version conflict**: The openpi-client pins numpy to 1.x; this is handled automatically by uv.
