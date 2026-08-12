# ArtiMo physical delivery contract

All values in this contract are measured diagnostics. None is an execution or
export gate: the harness always runs every declared stage, writes the complete
physical video, and publishes the three files. A false value remains visible in
`result.json` but never authorizes early termination or omission of output.

## Frozen inputs

Freeze the task JSON, source and optional physics URDF, every transitive mesh /
material / texture, ArtiMo plan, optional first-frame trajectory, robot URDF,
generic schemas,
harness code, tool versions, model, and reasoning effort. Never put credentials
in prompts or artifacts.

The generic harness code and all frozen inputs must be byte-identical between
handoff and release. Per-task poses, gains, mechanisms, camera, and seeds are
execution data stored in `grasp.json` and debug evidence.

A collision-only URDF generated after handoff is execution data, not a source
input or config. Store it only below the current `.artimo-runs/<task_id>/`,
record its path in `grasp.json` root `physics_urdf`, verify its mechanism against
the locked source URDF, and hash the exact simulated file in both clean runs.

The trajectory is optional and is never an animation source. If supplied, only
its first `joint_angles` row establishes a non-zero initial object state. With
no trajectory, initialize every URDF joint at its default zero state.

## Physical rollout

- Initialize object state once; subsequent object motion comes from rigid
  contact or a disclosed measured-state causal rule.
- Hold every movable object joint that is absent from plan drivers, declared
  passive returns, and causal effects at its initial state. Report its maximum
  displacement in both rollouts and require it to remain within `1e-4` m or
  rad; an unowned joint is not an animation channel.
- Require one exhaustive `control_execution` owner per plan timeline control;
  mixed-owner controls in the same phase remain separate. Every
  `robot_contact` control has its own contact stage and positive measured target
  contact; no causal actuator may command its joint target. Consecutive stages
  before any intervening plan `control_release` use the same contact link and
  one `contact_sequence`; `causal.json` is not an action-authority input. Preserve
  the same grasp without finger opening, retreat, constraint recreation, or a
  sample-zero IK resolve between plan phases. A link change is the normal
  release/reacquire boundary; same-link release needs an explicit plan-semantic
  boundary such as passive return.
- Execute the agent-declared contact acquisition exactly. `open_then_close`
  keeps the gripper open through transit/approach, closes while the arm holds
  the first contact pose, settles, manipulates, then opens before retreat.
  `maintain_width` keeps one declared width throughout contact. Record the
  invariant 200 N per-finger servo force in the serialized robot commands.
  When a physical push selects one fingertip/tool surface, record its
  robot-EEF-frame surface vector as `robot_tool_contact_offset_eef_m`; the
  declared object contact point remains on the object surface. Planning and
  rollout consume the same offset, and every unselected robot link remains
  collision-forbidden. A bounded tool-face search must declare its collision
  face centroid, edge midpoints, and vertices before numerical evaluation; a
  one-axis slice or result-driven candidate expansion is not accepted.
- Orientation images for a one-tool `physical_push` are judged by the nominated
  tool surface covering the intended button along its surface normal without
  sweeping the housing, not by opposing-jaw straddle. Switching from centered
  grasptarget geometry to a tool offset invalidates prior visual decisions and
  requires a fresh full eight-roll visual-only gate before placement.
- Keep robot/object collision response enabled except nominated contact pairs
  in the hidden negative control.
- Record every robot/object contact pair, constraint, joint reset,
  joint state, robot command, and causal-state transition each simulation step.
- For each causal rule, record first target contact, trigger-driver first
  motion, latch, effect enable, and effect-joint first motion ticks. Require the
  driver first motion to be no earlier than target contact, and effect enable
  and first motion to be strictly later than target contact.
- A gravity-loaded passive return uses execution-data force no smaller than the
  absolute inverse-dynamics generalized load at its reached state times a
  disclosed generic safety factor, and no larger than the source URDF effort
  limit. Verify the endpoint in the physical trace; duration cannot substitute
  for insufficient force.
- Require every joint requested by the ArtiMo plan to reach the task's minimum
  motion ratio at the correct point in the requested order, including temporary
  extrema for joints that later return.
- Non-target and indirect-effect-link robot contacts equal zero.
- Before rollout, the complete robot including its fixed base and physical support clears every
  declared forbidden moving link over home-to-approach, approach, manipulation,
  release/retreat, and every transit actually present in the command schedule,
  with prior stage endpoints preserved. A release-before-passive-return boundary
  holds at its safe retreat waypoint and does not imply a return home. An
  ordinary change to a different robot contact also transits directly from the
  released retreat endpoint to the next precontact path; it must not insert a
  retreat-to-home and home-to-next-stage detour.
  A release solver evaluates the full authoritative object state immediately
  before its boundary, including completed internal effects and holds, rather
  than only preceding robot-stage driver endpoints.
  If that direct segment is obstructed by geometry moved during an earlier
  stage, the incoming stage uses ordered world-frame transit waypoints. Planning
  clearance and rollout both consume the same serialized waypoint joint path;
  the waypoints affect only this incoming transit and are not reversed during
  retreat. Generate the bounded table once with
  `applications/artimo_robot_contact/propose_artimo_transit_routes.py`, using the moved state and the
  smallest expanded forbidden-link AABB intersecting the direct Cartesian
  segment. Evaluate that immutable table in one
  `applications/artimo_robot_contact/solve_artimo_transit_clearance.py --jobs 4` invocation. Every
  candidate runs in an isolated process, first with direct interpolation and
  then, when needed, bounded deterministic joint-space RRT. Planning and
  rollout consume the exact same serialized accepted joint path.
  An
  `open_then_close` stage has measured near-target
  geometry for the required gripper links at every sampled pose; EEF IK alone
  is not grasp evidence. Near-target geometry is only a cheap precheck. At the
  closed aperture, PyBullet must report simultaneous target contact from every
  declared `allowed_robot_contact_links` member; a positive gap or one-finger
  contact cannot trigger the disclosed ideal attachment.
- On the declared target contact link, collision allowance applies only to the
  declared `allowed_robot_contact_links`. Every other robot link plus the fixed
  support remains a non-target collision and is checked over the same dense
  scheduled path.
- Record maximum adjacent commanded joint motion, minimum joint-limit margin
  with its stage sample/joint, maximum actual joint step, and maximum
  commanded-versus-actual tracking error. Hard-limit saturation remains a
  failed diagnostic even when the local trajectory is finite.
- Target contact is present for the required continuous duration. Contact force
  is neither measured nor used as an acceptance or causal-trigger input.
- A passive-return motor is disabled before its plan-owned phase and enabled
  only from that phase onward; it may not resist an earlier robot-contact phase.
  Any internal-mechanism motion triggered by robot contact is likewise held
  until its plan-owned phase and until the triggering robot has completed a
  required release retreat. When dependent mechanism motion or a passive return
  follows robot contact, execution explicitly releases, completes a
  schema-declared clearance retreat, and spends a fixed nonzero interval at the
  safe endpoint before enabling object motion. A declared
  world-frame release route begins at the exact final grasp command, replaces
  the default link-normal retreat, and is serialized as the same dense joint
  path used by planning and rollout. The full robot,
  fixed base, and support satisfy `minimum_release_swept_clearance_m` throughout
  retreat, endpoint settle, and every sampled later mechanism/return state.
- Every `open_then_close` stage uses one disclosed ideal fixed constraint after
  physical acquisition and removes it only at the explicit release boundary.
  The acquisition tick must contain simultaneous target contact from every
  declared allowed robot contact link.
  The output states that it tests contact-pose/IK trajectory feasibility, not
  frictional force closure. `maintain_width` physical pushes create no fixed
  constraint.
- A plan-owned `hold_position` immediately following robot contact on the same
  driver joint retains the acquired grasp and endpoint. A terminal hold remains
  attached through the final settle and rendered frames; simulator cleanup is
  not a visible or semantic release. No release, retreat, or home transit may
  precede the declared hold.

An internal effect actuator is permitted only when execution data assigns that
specific phase to `internal_mechanism`, supplies an auditable physical
trigger/transmission justification and explicit `energy_source`, and the source URDF lacks the cross-joint
transmission.
Enable it no earlier than its authoritative `source_effect_phase`, and only
after measured target control displacement plus contact dwell and clearance;
bound its force/torque and derive targets from ArtiMo endpoints.

## Negative control

Run the identical serialized robot command schedule and deterministic seeds,
disabling only nominated target contact pairs. It must show zero target contact,
no causal trigger, and no object-joint motion beyond initial tolerance. Do
not place this rollout in the public video.

## Visual evidence

Before placement is frozen, retain four separate full-resolution view files for
every wrist roll generated by
`applications/artimo_robot_contact/render_artimo_grasp_orientation_candidates.py`; comparison sheets and
tiled multi-view cards are forbidden. Every candidate keeps the same contact
point, surface normal, and grasp depth. Record a `valid`/`invalid` visual decision
and reason for every candidate. Apply those decisions through
`applications/artimo_robot_contact/apply_artimo_grasp_orientation_decisions.py`. The render pass uses only a
kinematic-free parallel-jaw proxy and must prove that it ran no IK. Visual-
invalid rolls are removed before placement and never enter an IK call,
trajectory, transit, or rollout. Give every visual-valid roll a unique
contiguous visual priority starting at 1. The decision gate itself runs no IK.
Placement gives the priority-1 roll its complete bounded base search and dense
path validation first; a lower-priority roll receives its first IK call only
after every allowed placement for all preceding rolls fails. A sparse
template-base orientation IK pass is not acceptance evidence. `open_then_close` candidates
also require bilateral contact, while `maintain_width` physical pushes prove
actual contact/dwell in rollout. One decision covers every consecutive
stage in the same `contact_sequence`: all members receive the same rotation,
only the acquisition stage is probed, and full placement validates the inherited
arm reference throughout the complete sequence. The final chained
orientation gate covers every robot-contact stage and hashes the byte-identical execution supplied to
placement. Candidate previews are planning evidence and never appear in the
physical rollout video.

`video.mp4` is H.264, decodes completely, and shows only the physical rollout.
Keep robot, object, target contact, and resulting motion visible. A crop must be
from the same simulator frame. Review at least the configured fps over the whole
video for contact visibility, target completion, collision/interpenetration,
camera discontinuity, stale/black imagery, and rendering artifacts.

## Final artifacts

The final directory contains exactly:

```text
video.mp4
grasp.json
result.json
```

`grasp.json` contains the complete accepted execution plan using link-relative
metre translations and XYZW quaternions. `result.json` preserves native traces
or trace hashes and contains this measured manifest shape:

```json
{
  "evidence": {
    "schema_version": 2,
    "task_spec_sha256": "64 lowercase hex chars",
    "handoff_lock_sha256": "64 lowercase hex chars",
    "release_lock_sha256": "64 lowercase hex chars",
    "execution_plan_sha256": "64 lowercase hex chars",
    "physics_engine": "PyBullet",
    "physical_only_video": true,
    "object_trajectory_replay": false,
    "object_joint_resets_after_initialization": 0,
    "fixed_constraint_count": 0,
    "robot_command_schedule_sha256": "64 lowercase hex chars",
    "seeds": {"search": 0, "physics": 0},
    "physical": {
      "contacts": [
        {
          "stage_id": "stage identifier",
          "target_contact_observations": 1,
          "non_target_contact_observations": 0,
          "effect_link_contact_observations": 0,
          "maximum_driver_displacement": 0.0,
          "continuous_contact_s": 0.0
        }
      ],
      "joint_motion": {
        "joint name": {
          "requested_extrema": [0.0],
          "observed_extrema": [0.0],
          "minimum_progress_ratio": 1.0,
          "order_passed": true
        }
      }
    },
    "negative_control": {
      "same_robot_command_schedule": true,
      "target_contact_observations": 0,
      "causal_triggers": 0,
      "requested_joint_motion_remained_initial": true
    },
    "visual_qa": {
      "sample_rate_fps": 5.0,
      "no_visible_interpenetration": true,
      "physical_contact_visible": true,
      "requested_motion_visible": true,
      "no_rendering_artifacts": true
    }
  }
}
```

All values come from simulator/QA evidence, never schema-filling guesses. Video
creation and diagnostic success are separate: export always occurs, but
`passed` is true only when the video exists and every measured diagnostic
passes.
