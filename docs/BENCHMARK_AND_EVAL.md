# Benchmark and Evaluation

This guide covers the released benchmark package. The common use case is:
prepare the same assets as the benchmark, put your predictions under a
prediction root, then run 3D and optional 2D evaluation.

## Files

Set `BENCH` to the downloaded benchmark package, for example the
`benchmark_release/` folder from HuggingFace:

```bash
export BENCH=/path/to/benchmark_release
```

The package already contains the evaluation files:

```text
$BENCH/manifests/eval_manifest_225.json
$BENCH/manifests/phase_static_manifest.csv
$BENCH/manifests/asset_source_manifest.csv
$BENCH/annotations/
$BENCH/gt_animations/
$BENCH/gt_phase_static_meshes/
```

Normal users do not need to regenerate these manifests.

## Prepare Assets

Download the source datasets:

- PartNet-Mobility: https://sapien.ucsd.edu/browse
- ArtVIP: https://huggingface.co/datasets/X-Humanoid/ArtVIP/tree/main
- Lightwheel: https://github.com/LightwheelAI/Lightwheel-simready-asset?tab=readme-ov-file

Then create ArtiMo-format assets from the release manifest. `DATA` is the output
folder created by this step.

```bash
python tools/prepare_benchmark_assets_from_manifest.py \
  --asset_manifest "$BENCH/manifests/asset_source_manifest.csv" \
  --out_data_root /path/to/prepared_artimo_data \
  --source_root PartNet-Mobility=/path/to/partnet-mobility-v0.zip \
  --source_root ArtVIP=/path/to/artvip \
  --source_root Lightwheel=/path/to/Lightwheel_OpenSource.zip \
  --copy_mode symlink

export DATA=/path/to/prepared_artimo_data
```

The manifest decides whether each asset goes into `causal_data/` or
`not_causal_data/`; users do not need to label this manually. The output layout
is:

```text
$DATA/
  causal_data/<asset_name>/mobility.urdf
  not_causal_data/<asset_name>/mobility.urdf
```

PartNet source ids are mapped to benchmark names, e.g. raw PartNet `102186`
becomes `$DATA/causal_data/bin1/`. If a local PartNet folder is named with a
category suffix such as `102186_trashcan`, it is also accepted.

If you need textured GLBs for running ArtiMo on these assets, build them after
preparation:

```bash
JOBS=6 bash scripts/textured.sh --root "$DATA/causal_data" --all
JOBS=6 bash scripts/textured.sh --root "$DATA/not_causal_data" --all
```

This writes `animated_textured_<asset_name>.glb` into each asset folder. Existing
prediction evaluation does not usually require this file.

## Prediction Layout

Place each method output under one prediction root. The evaluator accepts common
layouts such as:

```text
<pred_root>/<class>/<asset>/<action>/plan_animated.glb
<pred_root>/<class>/<asset>/<action>/trajectory.jsonl
<pred_root>/<asset>/<action>/<asset>/plan_animated.glb
<pred_root>/<asset>/<action>/<asset>/trajectory.jsonl
```

Use `plan_animated.glb` for GLB predictions, `trajectory.jsonl` for joint/state
predictions, or both.

## 3D Evaluation

Run:

```bash
scripts/run_3d_eval.sh \
  --cases_manifest "$BENCH/manifests/eval_manifest_225.json" \
  --gt_phase_static_manifest "$BENCH/manifests/phase_static_manifest.csv" \
  --data_root "$DATA" \
  --prediction_root /path/to/pred_root \
  --prediction_variant full_agent \
  --out_dir outputs/eval_3d_full_agent \
  --variants full_agent \
  --sequence_source auto \
  --disable_terminal_state_check \
  --num_points_per_link 2048 \
  --voxel_resolution 32 \
  --pc_backend numpy
```

Metrics:

- `PN_gIoU`: part-normalized 3D AABB generalized IoU.
- `PN_PC`: point consistency from Chamfer distance, normalized by link scale.
- `PN_OC`: voxel occupancy F1.

For exact comparison with the released ArtiMo/Causal Agent numbers, keep
`--voxel_resolution 32` and `--pc_backend numpy`. PyTorch3D can accelerate
`PN_PC`, but small numeric differences are expected:

```bash
scripts/run_3d_eval.sh ... --pc_backend pytorch3d --gpu_devices 0
```

PyTorch3D install instructions:
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

If a method exports the correct motion but uses a different mesh scale, add:

```bash
scripts/run_3d_eval.sh ... --geometry_normalization shape_scale_to_gt
```

The diagnose output includes `geometry_initial_pred_to_gt_scale_ratio` to help
catch scale mismatches.

## 2D Evaluation

Run 3D evaluation first. The 2D evaluator reuses the 3D matching output:

```bash
scripts/run_2d_eval.sh \
  --manifest "$BENCH/manifests/eval_manifest_225.json" \
  --own_3d_dir outputs/eval_3d_full_agent \
  --out_dir outputs/eval_2d_full_agent \
  --data_root "$DATA" \
  --variants full_agent \
  --project_root "$BENCH" \
  --mesh_alignment scale_translate_3d \
  --no_terminal \
  --workers 4
```

Metrics:

- `P_MaskIoU`: per-part visible mask IoU.
- `P_BoundaryF1`: per-part silhouette boundary F1.
- `P_ContourCD`: normalized per-part contour Chamfer distance.

`--project_root` is used to find benchmark-side view metadata such as
`puppet_master_noncausal/reference_view_distances.json` when present.

## External 4D Baselines

For single-mesh animation baselines such as AnimateAnyMesh/AAM or Animate3D,
use `evaluation/evaluate_aam_3d_batch.py`:

```bash
python evaluation/evaluate_aam_3d_batch.py \
  --cases_manifest "$BENCH/manifests/eval_manifest_225.json" \
  --data_root "$DATA" \
  --pred_dir /path/to/fbx_or_glb_predictions \
  --out_dir outputs/eval_3d_animate3d \
  --variant_name animate3d \
  --gt_phase_static_manifest "$BENCH/manifests/phase_static_manifest.csv" \
  --alignment_mode scale_translate_3d \
  --assignment_mode component \
  --motion_scale_mode scale_motion \
  --disable_terminal_state_check \
  --num_points_per_link 2048 \
  --voxel_resolution 32
```

This aligns the exported mesh to the benchmark asset frame and records
`alignment_mode`, `alignment_scale`, and scale diagnostics in `aam_metrics.csv`.

## Optional: Regenerate GT

The release package already ships GT plans, trajectories, animated GLBs, and
phase-static GLBs. Only regenerate them if you are editing annotations.

Compile an annotation to a plan:

```bash
scripts/benchmark_annotation_to_plan.sh \
  --annotation_json /path/to/benchmark_case.json \
  --out_plan_json /path/to/compiled/plan.json \
  --fps 30
```

Execute the plan:

```bash
python tools/run_plan.py \
  --asset_root /path/to/asset_root \
  --plan_json /path/to/compiled/plan.json \
  --out /path/to/compiled/animation \
  --trajectory_npz /path/to/compiled/animation/trajectory.npz \
  --trajectory_jsonl /path/to/compiled/animation/trajectory.jsonl \
  --export_animated_glb \
  --use_glb_scene auto
```

Regenerate phase endpoint GLBs:

```bash
scripts/export_benchmark_phase_static_meshes.sh \
  --cases_manifest /path/to/cases_manifest.json \
  --out_dir /path/to/compiled_phase_static_meshes \
  --phase_source plan \
  --overwrite
```
