# ArtiMo robot-contact repository rules

For any URDF + ArtiMo plan (and optional first-frame trajectory) to robot-video task,
read every required document under `applications/artimo_robot_contact/docs/`
and use the single generic path:

```text
applications/artimo_robot_contact/run_artimo_robot_pipeline.py
  -> applications/artimo_robot_contact/run_agent_task.py
  -> applications/artimo_robot_contact/run_artimo_physics.py
```

- No asset registry, named profile, task-specific backend, stored grasp, or
  Python branch keyed by asset/task identity may participate.
- `plan.json` is authoritative for object-side phase names, controlled joints,
  targets, returns, holds, and ordering. Do not reconstruct those semantics
  from URDF geometry; URDF is used only to solve the robot's declared contact
  pose and execute the plan.
- `trajectory.jsonl` is optional and never replayed; when supplied, only its
  first frame initializes non-zero object joints. Otherwise use URDF zero state.
- Per-task link names, poses, robot base, gains, causal rules, camera, search
  bounds, and seeds belong only in schema-validated execution data.
- A task run must not edit generic harness code, schemas, skills, requirements,
  or source inputs. The runner rejects workflow hash changes.
- Physical causality is the deliverable: no animation replay, runtime object
  resets, global collision disable, or undisclosed attachment.
- Hidden negative control uses byte-identical robot commands with only nominated
  target contact disabled.
- Publish exactly `video.mp4`, `grasp.json`, and `result.json`; keep search and
  debug evidence under `.artimo-runs/`.
- If the generic engine lacks an interaction primitive, report that generic gap
  instead of creating an asset-specific tool.
