# Benchmark and Evaluation Usage

This document starts from the common release use case: you already have a
method prediction and want to compare it against the benchmark.

## Evaluate A Prediction

If the benchmark package was downloaded from HuggingFace, keep it anywhere and
set `BENCH` to that folder:

```bash
export BENCH=/path/to/benchmark_release
```

Download the source datasets from:

- PartNet-Mobility: https://sapien.ucsd.edu/browse
- ArtVIP: https://huggingface.co/datasets/X-Humanoid/ArtVIP/tree/main
- Lightwheel sim-ready assets: https://github.com/LightwheelAI/Lightwheel-simready-asset?tab=readme-ov-file

Then point the preparation script at those downloaded source roots. PartNet can
be passed either as the extracted dataset directory or as the downloaded zip.
ArtVIP and Lightwheel can be passed as raw USD datasets; the script converts
only the benchmark assets it needs into ArtiMo's URDF format.

The benchmark package already ships the evaluation manifests under
`$BENCH/manifests/`; normal users do not need to generate them.

The asset folders are named with ArtiMo benchmark names, not raw dataset ids.
The manifest maps each raw dataset asset to the benchmark folder name and split:

```text
asset_name,asset_collection,source_dataset,source_asset,source_asset_dir,source_file
bin1,causal_data,PartNet-Mobility,102186,102186_trashcan,PartNet-Mobility/dataset/102186
microwave1,causal_data,ArtVIP,microwave_door_13,,ArtVIP/Articulated_objects/small_appliances/microwave/microwave_1/model_microwave_1.usd
electric_kettle1,causal_data,Lightwheel,electric_kettle1,,Lightwheel/Lightwheel_OpenSource/Manipulation/ElectricKettle001/ElectricKettle001.usd
```

Choose an output directory for the prepared ArtiMo data, then generate it by
providing one source root per downloaded dataset. The script reads
`asset_collection` from the manifest and automatically writes to
`causal_data/...` or `not_causal_data/...` under that output directory; users
do not decide whether an asset is causal or non-causal:

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

Raw ArtVIP/Lightwheel USD conversion uses the current Python environment. Make
sure the repository requirements are installed first; the conversion path needs
`usd-core`/`pxr`, and 2D evaluation needs `opencv-python`. If you already
converted those datasets into folders containing `mobility.urdf`, pass those
converted roots instead.

After preparation, data has this layout:

```text
$DATA/
  causal_data/
    bin1/
      mobility.urdf
    microwave1/
      mobility.urdf
  not_causal_data/
    <asset_name>/
      mobility.urdf
```

Each prepared asset path is:

```text
$DATA/<asset_collection>/<asset_name>/mobility.urdf
```

For example, the original PartNet asset `102186` becomes
`$DATA/causal_data/bin1/`. If your local PartNet copy was renamed to include
the category, `102186_trashcan` is also accepted.

Before running ArtiMo's `run_agent` on one of these prepared assets, build its
textured animated GLB from the URDF and meshes:

```bash
bash scripts/textured.sh --root "$DATA/causal_data" bin1
```

To build textured meshes for every prepared benchmark asset, run the batch
command on both asset collections:

```bash
JOBS=6 bash scripts/textured.sh --root "$DATA/causal_data" --all
JOBS=6 bash scripts/textured.sh --root "$DATA/not_causal_data" --all
```

This writes `animated_textured_<asset_name>.glb` and
`animated_textured_<asset_name>.report.json` into each asset folder. For pure
benchmark evaluation of existing prediction GLBs/trajectories, this file is not
normally required unless the evaluator needs to fall back to asset rendering.

For a 3D prediction method, place each prediction as `plan_animated.glb` and/or
`trajectory.jsonl` under one of these common layouts:

```text
<pred_root>/<class>/<asset>/<action>/plan_animated.glb
<pred_root>/<class>/<asset>/<action>/trajectory.jsonl
<pred_root>/<asset>/<action>/<asset>/plan_animated.glb
<pred_root>/<asset>/<action>/<asset>/trajectory.jsonl
```

Run 3D evaluation:

```bash
scripts/run_3d_eval.sh \
  --cases_manifest "$BENCH/manifests/eval_manifest_225.json" \
  --gt_phase_static_manifest "$BENCH/manifests/phase_static_manifest.csv" \
  --data_root "$DATA" \
  --prediction_root /path/to/pred_root \
  --prediction_variant full_agent \
  --out_dir outputs/eval_3d_full_agent \
  --variants full_agent \
  --sequence_source auto
```

Main 3D metrics:

- `PN_gIoU`: part-normalized 3D AABB generalized IoU.
- `PN_PC`: Chamfer-distance-based point consistency similarity.
- `PN_OC`: voxel occupancy F1.

PyTorch3D is optional but useful in two places: the ArtiMo pipeline can use it
for GPU rasterization when preparing visual prompts/diagnostic renders, and 3D
evaluation can use it to accelerate `PN_PC`. For final runs, install PyTorch3D
from the official instructions:
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md. Then use:

```bash
scripts/run_3d_eval.sh ... \
  --pc_backend pytorch3d \
  --gpu_devices 0
```

`--pc_backend auto --gpu_devices 0` also uses PyTorch3D when available. If
PyTorch3D is not installed, use `--pc_backend torch --gpu_devices 0` for the
Torch CUDA backend or `--pc_backend numpy` for CPU-only evaluation.

For the agent pipeline, enable the GPU raster paths with:

```bash
export CODEX_TORCH_RASTER=1
export CODEX_PYTORCH3D_VIS_RASTER=1
export CODEX_TORCH_RASTER_ALLOW_CPU=1
```

For 2D evaluation of mesh/GLB predictions, run 3D evaluation first. The 2D
evaluator reuses the 3D matching output:

```bash
scripts/run_2d_eval.sh \
  --manifest "$BENCH/manifests/eval_manifest_225.json" \
  --own_3d_dir outputs/eval_3d_full_agent \
  --out_dir outputs/eval_2d_full_agent \
  --data_root "$DATA" \
  --variants full_agent \
  --project_root "$BENCH"
```

For a 2D GIF baseline, place GIFs under
`<puppet_root>/<asset>/<action>/reference_view_*.gif` or
`<puppet_root>/causal/<asset>/<action>/reference_view_*.gif`, then include
`--variants puppet_master --puppet_root /path/to/puppet_root`. It still needs
`--own_3d_dir` from a 3D eval run because GT endpoint frames and selected views
come from the benchmark matching step.

Main 2D metrics:

- `P_MaskIoU`: per-part visible mask IoU.
- `P_BoundaryF1`: per-part silhouette boundary F1.
- `P_ContourCD`: normalized per-part contour Chamfer distance.

## Manifest Fields

The release manifest is prebuilt and intentionally small. Each case records where the case
came from and where to find the GT artifacts:

```json
{
  "case_id": "casual_output:bin1:fully_open",
  "class": "bin",
  "asset_name": "bin1",
  "action_name": "fully_open",
  "action_prompt": "Fully open the trash bin",
  "source_dataset": "PartNet-Mobility",
  "source_asset": "102186",
  "source_asset_dir": "102186_trashcan",
  "source_file": "PartNet-Mobility/dataset/102186",
  "asset_collection": "causal_data",
  "annotation_path": "benchmark_release/annotations/bin_constraint_templates/cases/casual_output__bin1__fully_open.json",
  "gt_plan_json": "benchmark_release/gt_animations/bin/bin1/fully_open/plan.json",
  "gt_trajectory": "benchmark_release/gt_animations/bin/bin1/fully_open/animation/trajectory.jsonl",
  "gt_glb": "benchmark_release/gt_animations/bin/bin1/fully_open/animation/plan_animated.glb"
}
```

At eval time, `--data_root` resolves the local asset folder and
`--prediction_root` resolves method outputs. The evaluator writes the fully
resolved manifest to `outputs/eval_3d_full_agent/diagnose/resolved_manifest.json`.

## Batch Reproduction

To reproduce ArtiMo benchmark numbers, generate method outputs for all benchmark
cases first. Use a larger point/sample configuration for final 3D runs:

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
  --num_points_per_link 2048 \
  --voxel_resolution 64 \
  --pc_backend auto \
  --gpu_devices 0
```

Then run selected-view 2D evaluation:

```bash
scripts/run_2d_eval.sh \
  --manifest "$BENCH/manifests/eval_manifest_225.json" \
  --own_3d_dir outputs/eval_3d_full_agent \
  --out_dir outputs/eval_2d_full_agent \
  --data_root "$DATA" \
  --variants full_agent \
  --project_root "$BENCH" \
  --workers 4
```

## Optional GT Regeneration

The release package already includes GT plans, trajectories, animated GLBs, and
phase-static GLBs. If you need to regenerate them from annotations, use:

```bash
scripts/benchmark_annotation_to_plan.sh \
  --annotation_json /path/to/benchmark_case.json \
  --out_plan_json /path/to/compiled/plan.json \
  --fps 30
```

Then execute the plan:

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

To regenerate phase endpoint GLBs:

```bash
scripts/export_benchmark_phase_static_meshes.sh \
  --cases_manifest /path/to/cases_manifest.json \
  --out_dir /path/to/compiled_phase_static_meshes \
  --phase_source plan \
  --overwrite
```
