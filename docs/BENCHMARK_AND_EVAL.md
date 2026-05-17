# Benchmark and Evaluation Usage

This document describes how to use existing benchmark annotations/manifests with
ArtiMo. It does not describe the benchmark authoring process.

## Expected Benchmark Case Manifest

Most evaluation scripts expect a JSON manifest:

```json
{
  "cases": [
    {
      "case_id": "casual_output:bin1:fully_open",
      "class": "bin",
      "asset_name": "bin1",
      "action_name": "fully_open",
      "annotation_path": "/path/to/annotation.json",
      "asset_root": "/path/to/data/causal_data/bin1",
      "gt_trajectory": "/path/to/compiled_gt/trajectory.jsonl",
      "gt_glb": "/path/to/compiled_gt/plan_animated.glb",
      "gt_plan_json": "/path/to/compiled_gt/plan.json",
      "variants": {
        "full_agent": "/path/to/pred/trajectory.jsonl"
      },
      "variant_plans": {
        "full_agent": "/path/to/pred/plan.json"
      },
      "variant_glbs": {
        "full_agent": "/path/to/pred/plan_animated.glb"
      }
    }
  ]
}
```

Paths may be absolute or relative to the working directory.

## Compile Annotation to GT Plan

Use this when you have a benchmark annotation JSON and want an executable
`plan.json`:

```bash
scripts/benchmark_annotation_to_plan.sh \
  --annotation_json /path/to/benchmark_case.json \
  --out_plan_json /path/to/compiled/plan.json \
  --fps 30
```

If the annotation sidecar already points to a source plan and you want exact
source dynamics:

```bash
scripts/benchmark_annotation_to_plan.sh \
  --annotation_json /path/to/benchmark_case.json \
  --out_plan_json /path/to/compiled/plan.json \
  --preserve_source_dynamics
```

Then execute the compiled plan:

```bash
python tools/run_plan.py \
  --asset_root /path/to/asset_root \
  --plan_json /path/to/compiled/plan.json \
  --out /path/to/compiled/animation \
  --trajectory_npz /path/to/compiled/animation/trajectory.npz \
  --trajectory_jsonl /path/to/compiled/animation/trajectory.jsonl \
  --export_animated_glb \
  --use_glb_scene auto \
  --skip_frame_render
```

## Export Phase Endpoint Static GLBs

The 3D/2D evaluation can use per-phase static endpoint GLBs for cleaner GT
states:

```bash
scripts/export_benchmark_phase_static_meshes.sh \
  --cases_manifest /path/to/cases_manifest.json \
  --out_dir /path/to/compiled_phase_static_meshes \
  --phase_source plan \
  --overwrite
```

This writes:

- `phase_static_manifest.csv`
- `phase_static_manifest.json`
- per-case `phase_static_manifest.json`
- one GLB per phase endpoint.

## 3D Evaluation

Run:

```bash
scripts/run_3d_eval.sh \
  --cases_manifest /path/to/cases_manifest.json \
  --out_dir evaluation/3d_full_agent \
  --variants full_agent \
  --num_points_per_link 2048 \
  --voxel_resolution 64 \
  --dynamic_weight 0.8 \
  --static_weight 0.2
```

Main metrics:

- `PN_gIoU`: part-normalized 3D AABB generalized IoU.
- `PN_PC`: Chamfer-distance-based point consistency similarity.
- `PN_OC`: voxel occupancy F1.

Frame matching is ordered by benchmark phase. The matching score uses weighted
3D AABB gIoU, with default dynamic/static weights `0.8/0.2`.

## 2D Evaluation

Run after 3D matching directories exist:

```bash
scripts/run_2d_eval.sh \
  --manifest /path/to/cases_manifest.json \
  --own_3d_dir /path/to/full_agent_3d_dir \
  --aam_3d_dir /path/to/animate_anymesh_3d_dir \
  --animate3d_3d_dir /path/to/animate3d_3d_dir \
  --puppet_root /path/to/final_puppet_results \
  --out_dir evaluation/2d_selected_view
```

Main metrics:

- `P_MaskIoU`: per-part visible mask IoU.
- `P_BoundaryF1`: per-part silhouette boundary F1 with pixel tolerance.
- `P_ContourCD`: normalized per-part contour Chamfer distance.

PuppetMaster is evaluated in its selected input view. Mesh baselines are
rendered into the same selected view and crop before part-mask scoring.
