# ArtiMo

ArtiMo is an agentic pipeline for action-conditioned articulated-object
animation. Given an articulated asset and a user action, it generates causal
grounding, compiles an executable animation plan, refines the plan through visual
diagnosis, and exports trajectory files, animated GLB, and optional Isaac Sim
execution bundles.

This repository is self-contained at the source-code level. Run commands from
the `ArtiMo/` root so internal `tools/...` calls resolve locally.

## Which File Should I Use?

Start here for the common workflows:

| Task | Run this | Details |
| --- | --- | --- |
| Prepare a raw URDF+mesh asset | [scripts/textured.sh](scripts/textured.sh) | See [Input Asset Format](#input-asset-format). |
| Run ArtiMo on one action | [scripts/run_agent.sh](scripts/run_agent.sh) | See [Run the Full Agent](#run-the-full-agent). |
| Execute an existing `plan.json` | [scripts/validate_existing_plan_export.sh](scripts/validate_existing_plan_export.sh) | See [Execute an Existing Plan](#execute-an-existing-plan). |
| Export an Isaac Sim runner/video | [scripts/compile_isaacsim_bundle.sh](scripts/compile_isaacsim_bundle.sh) | See [Isaac Sim Export](#isaac-sim-export). |
| Evaluate 3D/2D predictions on the benchmark | [scripts/run_3d_eval.sh](scripts/run_3d_eval.sh), [scripts/run_2d_eval.sh](scripts/run_2d_eval.sh) | Use the prebuilt manifests in `benchmark_release/manifests/`; see [docs/BENCHMARK_AND_EVAL.md](docs/BENCHMARK_AND_EVAL.md). |
| Use the benchmark package from HuggingFace | `benchmark_release/README.md` in the downloaded package | It explains how to point this repo's eval scripts at the downloaded data. |

## Layout

```text
ArtiMo/
  tools/       Core agent/runtime implementation.
  evaluation/  2D/3D evaluation entry points and helpers.
  benchmark/   Benchmark annotation compilation and phase-static export tools.
  scripts/     Stable shell entry points.
  configs/     Environment template.
  docs/        Evaluation and benchmark usage notes.
  examples/    Place example assets/manifests here.
```

Main entry points:

- `tools/run_agent_single.py`: preprocessing -> VLM -> LLM plan -> optional loop -> GLB export.
- `tools/run_agent_loop.py`: coverage loop and motion-diagnosis refinement loop.
- `tools/run_plan.py`: execute an existing `plan.json` and export trajectory/GLB.
- `tools/compile_isaacsim_executor.py`: compile an Isaac Sim runner bundle.
- `evaluation/evaluate_ablation_3d.py`: 3D benchmark evaluation.
- `evaluation/evaluate_2d_part_final.py`: 2D selected-view benchmark evaluation.
- `benchmark/benchmark_annotation_to_plan.py`: compile benchmark annotations to executable plans.
- `benchmark/export_benchmark_phase_static_meshes.py`: export benchmark phase endpoint GLBs.

## Environment

Python 3.10+ is recommended.

```bash
conda create -n artimo python=3.10 -y
conda activate artimo
pip install -r requirements.txt
```

Configure API and rendering:

```bash
cp configs/env.example .env
# edit .env
set -a
source .env
set +a
```

Required for model calls:

- `OPENAI_API_KEY` or another OpenAI-compatible key, or
- `GEMINI_API_KEY` for Gemini-backed VLM calls.

Required for high-quality rendering:

- `BLENDER_BIN=/path/to/blender`

Optional for Isaac Sim:

- `ISAAC_PYTHON=/path/to/isaacsim/python.sh`

### Optional Accelerators

ArtiMo runs without these optional accelerators, but the full benchmark pipeline
is much faster with them.

**Blender rendering.** Install Blender from the official download page:
https://www.blender.org/download/. ArtiMo uses Blender in background mode for
reference and motion-diagnosis rendering. Point `BLENDER_BIN` at the executable:

```bash
export BLENDER_BIN=/path/to/blender
"$BLENDER_BIN" --background --version
```

Recommended rendering settings:

```bash
export CODEX_LOOP_MOTION_RENDER_BACKEND=blender
export CODEX_BLENDER_USE_GPU=1
export CODEX_BLENDER_PERSISTENT=1
export CODEX_BLENDER_DEVICE_TYPE=OPTIX   # or CUDA
export CODEX_BLENDER_GPU_INDEX=0
```

`CODEX_BLENDER_PERSISTENT=1` keeps a background Blender worker alive between
renders, which avoids repeatedly launching Blender during coverage/motion loops.
Set `CODEX_BLENDER_PERSISTENT=0` only when debugging Blender worker state.

**PyTorch3D / Torch GPU rasterization and evaluation.** Install PyTorch and
torchvision for your CUDA version first, then install PyTorch3D following the
official instructions:
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md. A common
source-build fallback is:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
python - <<'PY'
import torch
import pytorch3d
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("pytorch3d", pytorch3d.__version__)
PY
```

Use PyTorch3D for 3D benchmark point-consistency evaluation with:

```bash
scripts/run_3d_eval.sh ... \
  --pc_backend pytorch3d \
  --gpu_devices 0
```

`--pc_backend auto --gpu_devices 0` also selects PyTorch3D when it is installed;
without `--gpu_devices`, `auto` falls back to the NumPy backend. If PyTorch3D is
not available but PyTorch CUDA is, use `--pc_backend torch --gpu_devices 0`.

The agent pipeline also uses PyTorch3D/Torch for visual prompt and diagnostic
rasterization when available:

```bash
export CODEX_TORCH_RASTER=1
export CODEX_PYTORCH3D_VIS_RASTER=1
export CODEX_TORCH_RASTER_ALLOW_CPU=1
```

**Isaac Sim.** Isaac Sim is needed only for the Isaac export/video path. Install
it from NVIDIA's official Isaac Sim installation docs:
https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html. After
installation, either make `isaacsim` available on `PATH` or set:

```bash
export ISAAC_PYTHON=/path/to/isaacsim/python.sh
```

Then compile and run the bundle as described in [Isaac Sim Export](#isaac-sim-export).

## Before Running `run_agent`

`run_agent` does not start from a loose mesh file. For each asset, prepare one
asset folder with:

```text
asset_name/
  mobility.urdf                         # required: joints, links, limits, mesh references
  meshes_or_textured_objs/...           # required: mesh/material files referenced by the URDF
  images_or_textures/...                # required if the materials reference texture images
  animated_textured_<asset_name>.glb    # required before running the agent
  animated_textured_<asset_name>.report.json
```

The first three items come from the dataset conversion step. The textured
animated GLB is generated from `mobility.urdf` plus its referenced meshes:

```bash
bash scripts/textured.sh --root /path/to/data/causal_data asset_name
```

After this preprocessing, run:

```bash
scripts/run_agent.sh \
  --asset_root /path/to/data/causal_data/asset_name \
  --action_text "your action"
```

## Input Asset Format

ArtiMo expects one directory per articulated object. Raw dataset assets should
contain a URDF plus the mesh/material files referenced by that URDF:

```text
asset_name/
  mobility.urdf
  textured_objs/                          # or another mesh folder referenced by the URDF
    part_or_original_*.obj
    part_or_original_*.mtl                # if the OBJ uses materials
  images/                                 # texture images referenced by MTL files, if any
  user_prompt.txt                         # optional
```

The URDF provides the joints, link hierarchy, visual origins, mesh filenames,
and joint limits. Mesh paths are resolved relative to the URDF location, so the
folder names do not have to be exactly `textured_objs/` and `images/`; they only
need to match the paths written in `mobility.urdf` and the OBJ/MTL files.

Before running the agent, build the canonical textured GLB used by rendering:

```bash
bash scripts/textured.sh --root /path/to/data/causal_data asset_name
```

For example:

```bash
bash scripts/textured.sh --root /path/to/data/causal_data trolley2
```

By default this first canonicalizes URDF names, then calls
`tools/build_textured_animated_glb.py` in `urdf_preview` mode. Link names are
rewritten to compact dataset-style IDs (`link_0`, `link_1`, ...), preferring
trailing numeric IDs already present in source names. Joint names are rewritten
in URDF order to `joint_0`, `joint_1`, ... . The original URDF is backed up as
`mobility.urdf.original_names.bak`, and the name mapping is written to
`urdf_name_map.json`.

It writes:

```text
asset_name/
  animated_textured_<asset_name>.glb
  animated_textured_<asset_name>.report.json
  textured_<asset_name>.glb -> animated_textured_<asset_name>.glb
```

The agent then uses `mobility.urdf` for kinematics and
`animated_textured_<asset_name>.glb` for textured reference rendering and
animated GLB export.

To process every asset directory under a dataset root:

```bash
JOBS=6 bash scripts/textured.sh --root /path/to/data/causal_data --all
```

Useful build controls:

```bash
INITIAL_POSE_MODE=zeros            # zeros | prismatic_lower | lower
FRAMES_PER_JOINT=36
MAX_PREVIEW_JOINTS=8
CANONICALIZE_URDF_NAMES=1          # set 0 to keep original URDF names
CANONICALIZE_JOINT_NAMES=1         # set 0 to keep original joint names
PY_BIN=/path/to/python
```

The batch command writes `animated_textured_build_manifest.json` and
`animated_textured_build_summary.json` under the dataset root.

If you already have the canonical GLB and report, this preprocessing step can be
skipped.

## Run the Full Agent

```bash
scripts/run_agent.sh \
  --asset_root /path/to/data/causal_data/bin1 \
  --action_text "fully open the trash bin" \
  --out_root outputs/bin1_fully_open \
  --vlm_model gemini-3.1-pro-preview \
  --llm_model gpt-5.4 \
  --api_provider openai \
  --use_glb_scene auto
```

Outputs:

```text
outputs/bin1_fully_open/bin1/
  images/                 overlay/reference views
  output.json             raw VLM response
  causal.json             parsed causal graph
  plan.json               executable animation plan
  trajectory.npz
  trajectory.jsonl
  plan_animated.glb
```

Enable coverage and motion-diagnosis refinement:

```bash
scripts/run_agent.sh \
  --asset_root /path/to/data/causal_data/bin1 \
  --action_text "fully open the trash bin" \
  --out_root outputs/bin1_fully_open_loop \
  --enable_loop \
  --enable_coverage_loop \
  --enable_motion_loop \
  --coverage_max_iters 1 \
  --motion_max_iters 3 \
  --use_glb_scene auto
```

The final selected outputs are copied back to the asset output root. Loop audit
artifacts are stored under `outputs/.../<asset>/loop/`.

### Mask-Conditioned Input

You can provide external mask image(s) as multimodal target grounding. The mask
images are attached to the VLM as `MASK_*` inputs, before the automatically
rendered reference/overlay images. The same mask metadata is reused by the
coverage and motion-diagnosis loops.

```bash
CODEX_LOOP_MOTION_RENDER_BACKEND=blender \
CODEX_BLENDER_USE_GPU=1 \
python tools/run_agent_single.py \
  --asset_root /path/to/data/not_causal_data/7_drawers \
  --action_text "open the masked areas(shown in translucent green area) as shown on the input mask image only for 2 seconds and slowly close for 2 seconds" \
  --out_root outputs/masked_7_drawers \
  --vlm_model gemini-3.1-pro-preview \
  --llm_model gpt-5.4 \
  --input_masks /path/to/masks/7_drawer_mask.png \
  --enable_loop \
  --enable_motion_loop \
  --motion_max_iters 3 \
  --disable_numeric_verify
```

Multiple masks can be passed after `--input_masks`. When more than one mask is
provided, ArtiMo builds a labeled mask panel under
`<out_root>/<asset>/loop/mask_inputs/` for loop-time VLM checks. The initial VLM
call records the exact attachments in
`<out_root>/<asset>/prompts/vlm_input_manifest.json`.

## Execute an Existing Plan

```bash
scripts/validate_existing_plan_export.sh \
  /path/to/data/causal_data/bin1 \
  /path/to/plan.json \
  validation_outputs/bin1
```

This runs `tools/run_plan.py` and exports:

- `trajectory.npz`
- `trajectory.jsonl`
- `plan_animated.glb`

## Isaac Sim Export

ArtiMo can compile an existing `causal.json` + `plan.json` into a runnable
Isaac Sim export bundle. The bundle contains Python code that imports the URDF
as an Isaac articulation, applies the plan as joint/base controls, captures
rendered frames, encodes an MP4, and writes the Isaac runtime trajectory.

Compile the bundle:

```bash
scripts/compile_isaacsim_bundle.sh \
  --asset_root /path/to/data/causal_data/bin1 \
  --causal_json outputs/bin1/causal.json \
  --plan_json outputs/bin1/plan.json \
  --bundle_dir outputs/bin1/isaacsim_bundle
```

Run the bundle with Isaac Sim:

```bash
ISAAC_PYTHON=/path/to/isaacsim/python.sh \
  outputs/bin1/isaacsim_bundle/run_isaacsim_executor.sh
```

If `isaacsim` is available on `PATH`, `ISAAC_PYTHON` can be omitted:

```bash
outputs/bin1/isaacsim_bundle/run_isaacsim_executor.sh
```

The runner also accepts debug-friendly capture options:

```bash
outputs/bin1/isaacsim_bundle/run_isaacsim_executor.sh \
  --resolution 640 360 \
  --max_frames 30
```

Bundle contents:

- `run_isaacsim_executor.py`: thin bundle entry point.
- `isaacsim_timeline_executor.py`: Isaac Sim runtime that imports the URDF,
  applies the compiled joint timeline, and records frames, video, and trajectory
  files.
- `executor_spec.json`: normalized timeline, joint limits, camera, and output
  paths.
- `import_asset/`: sanitized URDF and mesh assets copied into the bundle.

Default outputs are written under `outputs/` inside the bundle:

```text
outputs/bin1/isaacsim_bundle/outputs/
  <asset>_isaac.mp4        # Isaac-rendered video
  frames/frame_*.png       # captured rendered frames
  trajectory.npz           # Isaac runtime joint/base trajectory
  trajectory.jsonl
  execution_report.json    # paths, frame count, fps, video_encoded flag
```

The animation is represented by joint controls in `executor_spec.json`, for
example `joint_position`, `joint_velocity`, `hold_position`, and optional base
velocity controls. During execution, Isaac Sim steps the articulation and the
recorded MP4 is encoded from the rendered frame sequence.

By default Isaac renders the imported URDF articulation directly, so the visible
animation follows the same joint controls as the exported trajectory. Use
`--render_visual_mode baked_glb` at compile time only when you explicitly want
to overlay a pre-baked textured GLB render visual:

```bash
scripts/compile_isaacsim_bundle.sh \
  --asset_root /path/to/data/causal_data/bin1 \
  --causal_json outputs/bin1/causal.json \
  --plan_json outputs/bin1/plan.json \
  --bundle_dir outputs/bin1/isaacsim_bundle_baked \
  --render_visual_mode baked_glb
```

## Benchmark and Evaluation

Use [docs/BENCHMARK_AND_EVAL.md](docs/BENCHMARK_AND_EVAL.md) for benchmark
evaluation. The shortest path is:

1. Download or place the benchmark package as `benchmark_release/`.
2. Use `benchmark_release/manifests/asset_source_manifest.csv` to prepare raw
   dataset assets under `data/<asset_collection>/<asset_name>/`. The
   `asset_collection` field decides `causal_data` vs. `not_causal_data`; users
   do not classify assets manually.
3. Run 3D eval with [scripts/run_3d_eval.sh](scripts/run_3d_eval.sh).
4. Run 2D eval with [scripts/run_2d_eval.sh](scripts/run_2d_eval.sh).

[tools/prepare_benchmark_assets_from_manifest.py](tools/prepare_benchmark_assets_from_manifest.py)
can copy or symlink source assets into the expected benchmark layout. For
PartNet-Mobility, the manifest uses the raw numeric id, with an optional
`source_asset_dir` for local copies renamed as `<id>_<category>`. Pass one
source root per downloaded/converted dataset, for example
`PartNet-Mobility=/path/to/partnet-mobility`,
`ArtVIP=/path/to/data_artvip_converted`, and
`Lightwheel=/path/to/data_lightwheel_converted`.

If the benchmark is downloaded from HuggingFace and lives outside this repo,
use its shipped `manifests/eval_manifest_225.json` directly. The manifest
builder in `tools/` is only for maintainers who change the release contents.

For source-tree ownership, see [docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md).

## Notes

- `tools/` is reserved for the agent/runtime path. Evaluation and benchmark
  utilities live under `evaluation/` and `benchmark/`.
- Do not commit `.env` or API keys.
- PyTorch3D and Isaac Sim are optional unless you use GPU point-cloud evaluation
  or Isaac Sim rendering/execution.
