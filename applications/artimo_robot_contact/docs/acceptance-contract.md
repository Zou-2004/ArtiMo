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

Collision representation is a locked input, not agent-authored execution data.
Ordinary runs use the source URDF meshes byte-for-byte; they must not silently
replace them with whole-mesh convex hulls or V-HACD parts. A task may supply a
separately locked `inputs.physics_urdf`, whose mechanism must match the source.
The agent never writes `physics_urdf` execution data, and the exact simulated
file is hashed in the result.

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
  on the same contact link before any intervening plan `control_release` use one
  `contact_sequence`; `causal.json` is not an action-authority input. Preserve
  that grasp without finger opening, retreat, constraint recreation, or a
  sample-zero IK resolve between plan phases. A change to a different physical
  contact link is itself the required robot release/retreat/reacquire boundary,
  even when the object-motion plan has no standalone release phase; same-link
  release still needs an explicit plan-semantic boundary such as passive return.
- Execute the agent-declared contact acquisition exactly. `open_then_close`
  keeps the gripper open through transit/approach, closes while the arm holds
  the first contact pose, settles, manipulates, then opens before retreat.
  Its approach aperture is the selected final grasp aperture plus 0.005 m total
  clearance (0.0025 m per finger), rather than the robot's maximum aperture.
  `maintain_width` keeps one declared width throughout contact. Record the
  invariant 20 N per-finger servo force in the serialized robot commands.
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
  requires a fresh full five-roll visual-only gate before placement.
- Keep robot/object collision response enabled except nominated contact pairs
  in the hidden negative control and, after a real bilateral gate passes, the
  verified fingers plus their nearest common rigid palm parent against only the
  nominated target link.
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
  `applications/artimo_robot_contact/solve_artimo_transit_clearance.py` invocation. Direct
  interpolation is tested first. When execution selects cuRobo, one persistent
  GPU worker uses MotionGen graph search/trajectory optimization and GPU world
  collision for blocked segments; candidate evaluation is intentionally
  serialized through that worker to avoid multiplying VRAM. PyBullet performs
  the final exact swept-path verification. CPU RRT is allowed only in explicit
  Bullet mode or when `allow_bullet_fallback` is true. Planning and rollout
  consume the same accepted joint path.
  For an `open_then_close` stage, the no-IK visual review selects only the
  wrist-angle class. EEF IK alone is not grasp evidence. The application owns
  depth search; dense placement requires both finger links within the target
  gap and rollout requires sustained opposed bilateral target contact with
  settled finger motion. Only after that gate, the object-joint target may
  follow the measured fractional dense-sample index along the current
  serialized IK path and pass it through the identical smoothstep used to
  generate the object targets during planning. This preserves exact
  robot/object correspondence; joint-space arc length and linear joint
  interpolation are not used as proxies. Elapsed time is never actuation
  authority: if robot progress stalls or leaves the path tolerance, object
  progress stalls as well. No runtime grasp constraint is created.
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
  retreat, endpoint settle, and every sampled mechanism/return state before
  the next robot-contact acquisition. Later manipulation is checked by its own
  block, and cross-contact motion is checked only as transit.
- Every `open_then_close` stage uses one disclosed contact-gated object-joint
  actuator after its mandatory runtime bilateral-contact dwell gate. Both
  application-owned finger links must contact the nominated object link from
  opposed sides while finger velocities are settled and closure has
  progressed. Once verified, the two fingers and their nearest common rigid
  palm parent stop responding only to the nominated target link so they cannot
  form a redundant solver loop with the task actuator; the remaining arm and
  every non-target object link stay collision-authoritative. The actuator target
  interpolates the authoritative ArtiMo joint endpoint by monotonic measured
  robot dense-sample progress along the current IK path. Require the measured
  object-joint tracking error to remain within the harness tolerance. A failed gate never enables object
  actuation, and the hidden negative control replays the identical robot
  commands with no gate transition. No runtime grasp constraint is created.
  During a multi-joint uninterrupted contact sequence, verification persists
  across its stages and completed drivers remain at their achieved targets
  until explicit release or a later declared controller takes ownership.
  The output states that it tests contact-pose/IK trajectory feasibility and
  contact-gated plan execution, not frictional force closure.
- A plan-owned `hold_position` immediately following robot contact on the same
  driver joint retains the verified gate, final arm pose, and object endpoint.
  A terminal hold remains active through the final settle and rendered frames;
  simulator cleanup is not a visible or semantic release. No release, retreat,
  or home transit may precede the declared hold.

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
point and surface normal. Record `angle_status` first and independently record
`contact_point_status=valid/adjust` for the rendered semantic translation and
visible jaw relationship. Except for a one-tool `physical_push`, `valid`
requires the target feature cross-section to appear inside the open gap between
the cyan and magenta contact pads, with the pads on opposite sides, in at least
one isolated view that clearly exposes the jaw-closing cross-section. Every
required view must be opened, but an occluded view does not veto `valid` when
another isolated view clearly proves the relationship. Feature-pad overlap or
same-side pads in the proving view, or the absence of any proving view, is
`adjust`. This is a
plausibility review; it does not require exact static proxy collision. An
angle-valid candidate whose point is shallow, deep, off-target, or lacks any
clear proving view
requires execution-data repair and a
fresh complete five-roll render; it may not be hidden as an invalid angle.
Record a final `valid`/`invalid` decision and reason for every candidate only
after the retained angle's offset passes. Apply those decisions through
`applications/artimo_robot_contact/apply_artimo_grasp_orientation_decisions.py`. The render pass uses only a
kinematic-free parallel-jaw proxy and must prove that it ran no IK. Visual-
invalid rolls are removed before placement and never enter an IK call,
trajectory, transit, or rollout. Give every visual-valid roll a unique
contiguous visual priority starting at 1. The decision gate itself runs no IK.
Placement treats that reviewed nominal contact geometry as frozen and retains
every visual-valid roll for joint whole-task search. It
combines one candidate per independent manipulation block with each bounded base,
projects the authoritative future object state before every block, and ranks the
result by its worst block. Visual priority is only a deterministic tie-break
after geometric feasibility; it may not commit the first block before later
blocks are checked. Placement constructs five representative object-side
fractions for every robot-contact stage and records the world AABB center of the
child link controlled by that stage's `driver_joint`, after all authoritative
prior plan controls. It averages samples per unique moving link and then equally
averages all moving-link centers into one whole-task reference. The initial base
is offset from that reference, not from the first contact or first moving link.
The complete declared distance/lateral/height/yaw/orientation grid is ordered
deterministically from its center outward and split into consecutive bounded GPU
batches. Failure scores cannot move the center, introduce new directions, or
reorder remaining cells. As soon as a batch contains a complete sparse path,
run its survivors through dense and release checks in candidate order. Stop only
when one candidate passes sparse, the complete dense path, and release, or after
the declared bounded grid is exhausted. Sparse GPU collision uses the actual object collision
meshes at each sampled state, not whole-link AABBs; dense validation and the
final rollout provide exact target-contact confirmation. A sparse
template-base orientation IK pass is not acceptance evidence. `open_then_close`
candidates require a visually plausible opposed-jaw relationship and numerical
confirmation during dense validation/final rollout, while `maintain_width` physical pushes prove
actual contact/dwell in rollout. One decision covers every consecutive
stage in the same `contact_sequence`: all members receive the same rotation,
only the acquisition stage is probed, and full placement validates the inherited
arm reference throughout the complete sequence. The final chained
orientation gate covers every robot-contact stage and hashes the byte-identical execution supplied to
placement. Candidate previews are planning evidence and never appear in the
physical rollout video.

Independent application-owned grasp-depth groups use deterministic coordinate
search, never a Cartesian depth product. Begin from the agent value as an
untrusted search center, expand symmetrically through the application-owned
one-dimensional depths, retain the nearest best dense
evidence, and then advance to the next failed group.
Within each bounded sparse GPU batch, all sparse survivors remain eligible for
serial dense verification in deterministic cost order. Dense stops on the first
full manipulation-plus-release pass; rejection of the first survivor cannot
discard later survivors, and an already-passing depth group is not enumerated
to repair a different failed contact sequence.

Placement append-flushes a diagnostic `progress.jsonl` containing the whole-task
reference, every center-out sparse batch, dense candidate, release, and terminal
transition. It is monitoring evidence only and cannot participate
in acceptance or change the declared bounds and gates.

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
