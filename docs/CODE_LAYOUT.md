# Code Layout

ArtiMo keeps source files separated by runtime role:

- `tools/`: core agent/runtime code. These files are used by the public agent
  pipeline, plan execution, rendering, diagnosis loops, GLB export, and Isaac
  Sim bundle export.
- `evaluation/`: benchmark evaluation code only. This includes 3D matching,
  external 4D mesh loading, and selected-view 2D part-mask scoring.
- `benchmark/`: benchmark utility code. These scripts compile benchmark
  annotations to executable plans and export per-phase static endpoint GLBs.
Stable shell entry points live in `scripts/` and should be preferred over
calling Python files directly.
