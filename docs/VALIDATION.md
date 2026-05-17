# Validation

The independent `/home/chunyu/ArtiMo` copy was validated against an existing
causal asset and existing `plan.json` by running ArtiMo's local
`tools/run_plan.py` through:

```bash
./scripts/validate_existing_plan_export.sh \
  <asset_root> \
  <existing_plan.json> \
  validation_outputs/bin1_fully_open
```

The exported `trajectory.npz` matched the original trajectory exactly for:

- `joint_angles`
- `joint_names`
- `base_translation`
- `base_rotation_xyzw`
- `time_s`

The validation also generated `trajectory.jsonl` and `plan_animated.glb`.
Temporary validation outputs were removed after checking.

