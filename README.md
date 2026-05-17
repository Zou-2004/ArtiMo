# ArtiMo

ArtiMo is an agentic pipeline for action-conditioned articulated-object
animation. Given an articulated asset and a user action, it generates causal
grounding, compiles an executable animation plan, refines the plan through visual
diagnosis, and exports trajectory files, animated GLB, and optional Isaac Sim
execution bundles.

This repository is self-contained at the source-code level. Run commands from
the `ArtiMo/` root so internal `tools/...` calls resolve locally.

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

## Asset Format

Each asset directory should contain:

```text
asset_name/
  mobility.urdf
  animated_textured_<asset_name>.glb
  animated_textured_<asset_name>.report.json
  user_prompt.txt                         # optional
  images/                                 # optional textures
```

The URDF provides joints and link hierarchy. The GLB provides the textured
assembled mesh used for rendering and animated GLB export.

## Run the Full Agent

```bash
scripts/run_agent.sh \
  --asset_root /path/to/data/causal_data/bin1 \
  --action_text "fully open the trash bin" \
  --out_root outputs/bin1_fully_open \
  --vlm_model gpt-5.4 \
  --llm_model gpt-5.4 \
  --api_provider openai \
  --use_glb_scene auto \
  --skip_plan_frames
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
  --use_glb_scene auto \
  --skip_plan_frames
```

The final selected outputs are copied back to the asset output root. Loop audit
artifacts are stored under `outputs/.../<asset>/loop/`.

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
- `plan.mp4`

For a faster trajectory/GLB-only check, set `SKIP_FRAME_RENDER=1`.

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

See [docs/BENCHMARK_AND_EVAL.md](docs/BENCHMARK_AND_EVAL.md). For source-tree
ownership, see [docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md).

In short:

- Compile benchmark annotation JSON to plan:
  `scripts/benchmark_annotation_to_plan.sh ...`
- Export phase endpoint GLBs:
  `scripts/export_benchmark_phase_static_meshes.sh ...`
- Run 3D evaluation:
  `scripts/run_3d_eval.sh ...`
- Run 2D evaluation:
  `scripts/run_2d_eval.sh ...`

## Notes

- `tools/` is reserved for the agent/runtime path. Evaluation and benchmark
  utilities live under `evaluation/` and `benchmark/`.
- Do not commit `.env` or API keys.
- PyTorch3D and Isaac Sim are optional unless you use GPU point-cloud evaluation
  or Isaac Sim rendering/execution.
- Use `--skip_plan_frames` for fast trajectory and GLB export without MP4/PNG
  frame rendering.
