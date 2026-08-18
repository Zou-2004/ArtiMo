# Application: URDF + plan to a physical robot rollout

This document describes the isolated robot-contact application. It accepts an
object URDF, an ArtiMo `plan.json`, and an optional initial-state trajectory,
then runs contact selection, collision-aware IK and placement, one causal
PyBullet rollout with its hidden negative control, video review, and delivery
verification.

## 1. Install

Use Python 3.10 and make sure `ffmpeg` and `ffprobe` are on `PATH`.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r applications/artimo_robot_contact/requirements.txt
ffmpeg -version
ffprobe -version
```

The robot-contact requirements do not install cuRobo. cuRobo is an optional
GPU IK and motion-planning backend. If it is unavailable, the application
automatically uses the PyBullet/Bullet backend instead; the workflow remains
functional but placement and dense IK can be substantially slower.

## Optional GPU IK backend (cuRobo)

To reproduce the GPU planning path, install cuRobo in a separate Python
environment with a CUDA-enabled PyTorch build, following the
[cuRobo installation instructions](https://curobo.org/get_started/1_install_instructions.html).
The cuRobo environment must be able to import `curobo`, `torch`, and the
matching CUDA runtime. Keep the CUDA, PyTorch, and cuRobo versions compatible;
the application does not download or install any of them automatically.

The cuRobo library installation itself is typically:

```bash
git clone https://github.com/NVlabs/curobo.git
cd curobo
python -m pip install -e . --no-build-isolation
```

Install a compatible CUDA-enabled PyTorch build in that environment first.
cuRobo documents Linux as the primary platform and Windows as experimental;
the CPU fallback remains available when the GPU environment is not usable.

On the reference Windows setup, the application discovers the optional
backend at:

```text
C:\ProgramData\miniforge3\envs\artimo-curobo\python.exe
```

If that executable is not present (or on a system with a different layout),
the application intentionally falls back to Bullet CPU IK. The Panda URDF and
meshes themselves are bundled in
`applications/artimo_robot_contact/assets/panda/`; they are not downloaded at
runtime.

## 2. Run the complete workflow

From the repository root, run this command. Replace only the two input paths:

```bash
python applications/artimo_robot_contact/run_artimo_robot_pipeline.py \
  --urdf path/to/object/mobility.urdf \
  --plan path/to/animation/plan.json \
  --agent-command 'codex exec --sandbox workspace-write --ask-for-approval never -'
```

The final directory contains exactly:

```text
outputs/robot_contact/<task-id>/
  video.mp4
  grasp.json
  result.json
```

The object URDF and every mesh/material/texture it references must be inside
the repository checkout. The Panda URDF and meshes are already included. The
agent reads the English instructions in
`applications/artimo_robot_contact/docs/`; the repository-level `docs/`
contains the application overview and main ArtiMo documentation.

To use a different CLI agent, replace only `--agent-command` with a command
that reads the prompt from standard input.

## Optional inputs and preparation

`trajectory.jsonl` is optional. It is never replayed as animation; when
supplied, only its first `joint_angles` row initializes non-zero object joints.
Add this argument to the complete command:

```bash
--trajectory path/to/animation/trajectory.jsonl
```

Without it, the rollout starts from the URDF default zero joint state.

Use `--prepare-only` instead of `--agent-command` to generate and inspect the
frozen English `prompt.md` and `input-lock.json` without running the agent,
physics, or video export.
