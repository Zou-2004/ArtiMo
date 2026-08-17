# ArtiMo asset-independent failure playbook

## 1. Diagnostic flags, never rollout stops

Flag a candidate for global robot/object collision disable, undeclared fixed
attachment, visible interpenetration, post-initialization object reset,
trajectory replay, time-only effect triggering, unequal physical/control robot
commands, contact on an undeclared object link, or success inferred from JSON
without dense video review. Also flag any missing/duplicate plan-control owner,
causal actuation of a `robot_contact` phase, robot-owned stage without positive
contact evidence, or release/re-approach inserted inside one declared continuous
contact sequence. These flags may rank planning candidates and remain in
`result.json`, but they never truncate the dense trajectory, skip later
phases, suppress video export, or make the runner return a failure status.

## 2. Diagnosis order

1. Robot command tracking under the invariant harness-owned stiff Panda
   controller: no gravity sag and the commanded precontact pose is actually
   reached; failure here is a shared-harness regression, never a task tuning
   dimension;
2. ArtiMo phase/control indices, endpoints, extrema, returns, order, and
   exhaustive `control_execution` ownership;
3. robot stages versus justified internal mechanisms, including uninterrupted
   `contact_sequence` boundaries;
4. declared control contact point, normal/wrist orientation, and approach;
5. whole-task contact-keyframe reference and its initial bounded base stance;
6. center-out placement cell, continuous waypoint IK, dense path result, and
   swept-path clearance;
7. release retreat and the full passive-return swept-volume clearance;
8. target contact pair, direction, travel, and dwell for every
   robot-owned stage;
9. causal-rule state and declared internal-effect targets;
10. every plan-requested joint-motion ratio;
11. negative control and physical-video QA.

## 3. Bounded repairs

Change one execution-data class at a time in this order: contact point, contact
orientation, precontact offset, center-out base search, intermediate
waypoint, then camera. Arm controller
gain/force and grasp physics are fixed by the harness and are never repair
variables. Keep object orientation and facing construction fixed while
tuning distance/lateral reach unless the declared contact surface itself was
wrong. Keep search bounds and rejected candidates in debug evidence.

The visual pass freezes only the hard-valid contact set; it does not choose one
grasp before placement. Merge uninterrupted contacts into manipulation blocks,
project every future plan state, and build the base initializer from world
centers of every unique plan-driven moving child link sampled across every
block. Average samples within each moving link first, then weight all moving
links equally so the first or most frequently sampled link cannot dominate.
Order the complete declared
contact-facing grid from that whole-task center outward and evaluate consecutive
bounded batches on GPU. Do not move the center, reorder later cells from failure
scores, or add failure-derived directions. Evaluate every visual-valid roll in
the same declared order. Do not repeatedly invoke the planner for individual candidates,
replay the complete object motion merely to score a wrist roll, or cycle back to
a frozen class after trying later classes. If the bounded batches cannot produce
a valid path, record the exact failed clearance/IK gate as a generic gap.

Elapsed time is never a repair variable or a stopping rule. Do not shrink a
batch, keep only a candidate that happened to finish first, skip a declared grid
cell, or convert a failed diagnostic row into rollout data because a
command is slow or an interactive tool window ends. Keep the same inputs and
wait, poll, resume, or rerun the byte-identical numerical command. A placement
result without an emitted `execution.json` cannot enter release planning or a
physical rollout.

Never author or infer the common contact frame from a quaternion tuple. The
application derives it from the collision-surface normal and principal tangent;
an input quaternion is ignored. Never infer the remaining wrist-roll choice
from a quaternion tuple, comparison sheet, tiled card,
or one occluded full-scene render. After choosing the semantic point, run
`applications/artimo_robot_contact/render_artimo_grasp_orientation_candidates.py` once for the complete
roll batch. Open all four separate files for every candidate. Use the isolated
surface-normal and tangent views with cyan/magenta opposing contact links to
verify that the jaws straddle the intended feature; do not accept a roll merely
because finger shafts appear parallel to a handle or rim. Mark every candidate
angle-valid or angle-invalid first. For every angle-valid row, independently
record `contact_point_status=valid/adjust`. A high, low, or off-target point, or
the absence of any isolated view that clearly proves the feature lies between
opposed pads, must be repaired in execution data and the complete five-roll
batch must be rerendered. An occluded view alone does not veto another clear
proving view. Never discard a correct angle to hide an incorrect offset. Only
then mark every candidate visual-valid or visual-invalid with a reason and apply the manifest through
`applications/artimo_robot_contact/apply_artimo_grasp_orientation_decisions.py`. Visual-invalid candidates
are not low-ranked candidates: the visual pass runs no IK, and invalid rolls are
absent before placement. IK and whole-arm clearance may reject only the
visual-valid set. Sparse and dense base search do not repeat finger-gap queries
for contact geometry already frozen by the visual offset gate, and rollout uses
the disclosed verified contact gate after closure rather than rechecking the depth. Rank that set first by visual semantics
with unique contiguous priorities, but use priority only as a deterministic
tie-break after joint whole-task geometry. Preserve all valid candidates for the
base-and-block contact combination search. Do not use a default/template
Panda base plus five sparse driver samples as a preliminary gate. Require a
visually plausible two-sided jaw relationship for an `open_then_close` grasp;
exact target contact is confirmed only by dense validation and rollout. Do not require it for a
`maintain_width` physical push, whose actual contact/dwell is measured
during rollout. For one uninterrupted `contact_sequence`, render and probe only
the first acquisition stage, apply its roll to every member, and validate the
inherited arm branch with full placement; never probe a later member
independently from home. If no
candidate has both the required visual relationship and
valid measurements, report the bounded orientation gap instead of manually
inventing another pose.

If a visually correct roll places the feature high, low, or outside the jaw
span, keep the angle decision, mark the point `adjust`, repair only the
task-local contact translation, and rerender all five rolls. Static proxy
collision is advisory rather than a hard gate. If the semantic point is
plausible but the grasp still looks shallow along the surface normal, keep the
angle decision and run placement's
application-owned depth search. Do not ask the agent to edit `grasp_depth_m`.
For centered Panda `open_then_close`, value zero includes the robot-defined
0.015 m inward useful-finger inset and equals an effective `-0.015 m`; the bounded
rule-based lattice applies continuous adjustments around that baseline and
orders them shallow-to-deep. The visual gate requires a
convincing opposed grasp but does not require physical collision/contact from
the no-IK proxy. Simultaneous numerical target contact from every declared
grasp link is confirmed only for the dense shortlist and final rollout. A
single allowed fingertip touching the target is not a grasp.

Dense IK continuation is harness-fixed to a 0.08-rad local trust region around
the preceding command. This prevents redundant-joint null-space wandering and
remote branch switches while preserving a finite best-local trajectory. A pose
residual at a joint limit remains diagnostic data; never remove the trust region,
restart global IK mid-trajectory, or stop the remaining rollout to reduce it.
The first IK configuration of an independent acquisition is not adjacent to
home or the preceding retreat in this sense. Its raw entry joint delta is a
diagnostic for the transit planner, not a placement rejection gate; the actual
approach/transit must be interpolated or routed with cuRobo MotionGen when the
GPU backend is selected; bounded CPU RRT is only Bullet mode or explicit fallback.
Read the reported minimum joint-limit margin and its stage sample/joint. If the
margin reaches the numerical boundary while Cartesian residual grows, classify
the candidate as a local-branch dead end. Do not hide it with a remote global
restart or a task-specific regularization weight; repair the earlier execution
data class or report the bounded generic gap.

Read the diagnostics before choosing a repair:

- Open the survey's isolated `contact_link_reference__*.png` and every listed
  `contact_feature_reference__*.png` before interpreting
  a numerically valid point. If the point lies on the declared link but lands on
  a broad door/lid/panel while the isolated view exposes a handle, button, rim,
  lip, or latch, reject it as a semantic-contact failure. Move the execution-data
  point to the visible control feature; do not change link ownership or add an
  asset-specific rule.

- `surface_outward_axis_world`, `grasp_target_world_m`, and
  `precontact_world_m` must place every positive precontact offset outside the object.
  Contact local +Z is outward and robot motion toward contact is -Z; never
  repair a reversed approach by making `precontact_offset_m` negative. The
  final `grasp_depth_m` is signed and application-selected around the centered Panda
  baseline (`effective offset = -0.015 m + grasp_depth_m` for centered
  `open_then_close`). The bounded solver selects it only after exact dense
  bilateral target-gap and collision validation. Panda grasptarget
  +Z runs from palm to fingertips and is converted by the harness to contact
  -Z. If the palm faces the object while fingertips point outward, repair the
  shared frame conversion, not the task quaternion.
- `precontact_offset_m` vanishes when manipulation starts, while
  `grasp_depth_m` remains. If IK succeeds but the gripper misses, read
  `target_link_gap_by_allowed_robot_link_m` and the runtime
  `grasp_acquisition` record; the rule-based search must reject that depth.
  Do not use a large grasp depth as approach clearance.
- If contact is a large, short impulse followed by a miss, inspect
  `contact_acquisition` before changing offset. A grasp target approached with
  `maintain_width` or with an already-closed aperture can strike the target end
  face without ever acquiring it. The generic `open_then_close` approach
  aperture is the selected final aperture plus 5 mm total clearance; do not
  replace it with a maximum-width opening that sweeps adjacent housing. Use
  `open_then_close` only when that aperture can surround the target; tune only close duration from measured
  contact/dwell. Finger servo force is a harness constant. For a button or direct push, deliberately choose
  `maintain_width` instead of inserting a fictitious grasp.
- If a button probe has valid IK but the centered closed gripper overlaps the
  surrounding body, do not move the declared object contact point off the
  button or accept the overlap. A centered-gripper jaw-straddle decision is not
  reusable for a one-tool push. Inspect the real robot collision geometry,
  render every wrist roll with the left/right collision-face centroid and no
  IK, and hard-gate by square tool coverage, declared-normal approach, and no
  visible housing sweep. For only those newly visual-valid roll/tool pairs,
  enumerate a closed set containing each plausible fingertip collision face's
  centroid, edge midpoints, and vertices only for rolls already marked
  visual-valid, and align the selected surface with
  `robot_tool_contact_offset_eef_m`. Keep the other finger, hand, and arm
  collision-forbidden. Write the complete candidate manifest before any probe;
  do not inspect one face axis and then expand adaptively. Freeze the
  least-offset feasible tool frame, rerender
  the final immutable visual batch, and only then form the placement gate.
- If an `open_then_close` stage passes bilateral verification but its object
  joint does not follow, inspect measured dense-sample robot-path progress,
  progress residual, task-actuator target, object tracking error, and motor
  ownership. Do not tune friction,
  finger force, arm force/gain, penetration, or elapsed-time gates. Timing
  repair with `manipulation_sample_hold_s` applies only to physical pushes.
- If an object drifts during a same-joint `hold_position` immediately after a
  robot-contact phase, inspect the serialized command schedule. A release,
  retreat, or home transit before that hold is a shared scheduler regression;
  preserve the acquired grasp through the hold instead of adding friction,
  damping, a task-specific motor, or a longer pre-release dwell.
- If an upstream driver refuses to move during a later stage of one
  uninterrupted grasp, inspect controller ownership and confirm the current
  task actuator is the final write before the physics step. The current object
  target must follow measured progress along that stage's robot path; completed
  drivers hold their achieved targets until explicit release.
- `robot_base_outward_halfspace_m` must be positive and
  `panda_eef_to_contact_inward_alignment_cosine` must be near +1 in the scene
  report. A negative half-space value means the arm is reaching around the
  object from behind even if endpoint IK succeeds; move the base or select the
  genuinely exposed surface before tuning IK.
- `physical.contact_diagnostics[].unexpected_contact_pairs` names the robot link,
  object link, phase, observation count, and deepest penetration for every
  non-target contact. Repair that pair.
- A collision between the declared target object link and a robot link absent
  from `allowed_robot_contact_links` is still a non-target collision. If the
  physical trace reports it but dense planning does not, treat that as a shared
  swept-clearance regression rather than widening the allowed contact set.
- Compare `robot_tracking.maximum_commanded_joint_step_rad` with
  `maximum_actual_joint_step_rad` and tracking error. A small commanded step
  with a large actual step/contact impulse is collision-driven snapping, not an
  IK branch jump; repair geometry or path clearance, never controller gains.
- `ik[].minimum_swept_clearance_m` and `ik[].tightest_swept_samples` show how
  close the path came to a forbidden link, which robot link (including the
  fixed base or `robot_support`) was tightest, and at which sample. Placement must be scored with
  prior-stage endpoints still applied; an initial-frame-only base check is not
  evidence for a moving door, lid, tray, or panel.
- A collision reported in an actually scheduled `transit_in` or `transit_out`
  invalidates the rest posture or fixed placement even when every manipulation
  sample is clear. Do not diagnose a nonexistent retreat-to-home after a
  release-before-passive boundary; that schedule releases, retreats to the
  declared safe waypoint, and holds there through the return.
- Between two different robot-contact stages, verify that the next transit
  starts at the preceding released retreat endpoint. A forced return to home is
  a shared scheduler regression when the plan asks for the next robot stage;
  repair the scheduler rather than moving the base around the manufactured
  detour. If the direct segment intersects an obstacle moved by the prior
  stage, run `applications/artimo_robot_contact/propose_artimo_transit_routes.py` against the frozen placed
  execution. A dense candidate whose only failing pair is such a moved-link
  `transit_in` collision must be marked
  `prior_plan_moved_link_blocks_transit` and sent to this bounded solver before
  rejection; it must not be promoted directly to rollout. It applies preceding endpoints, measures the smallest intersecting
  forbidden-link AABB, and emits four lateral plus three top-corner route
  candidates without knowing the asset. Evaluate that immutable batch once
  with `applications/artimo_robot_contact/solve_artimo_transit_clearance.py`. Direct joint
  interpolation is attempted first. With cuRobo selected, one persistent GPU
  MotionGen worker evaluates blocked segments and GPU collision worlds; exact
  PyBullet swept-clearance then verifies the returned path. CPU RRT may run only
  in Bullet mode or with explicit `allow_bullet_fallback: true`. The accepted path is used for
  rollout. Rank by full-robot swept clearance and path length, freeze the
  least-cost feasible route, and never replay it on retreat. If all candidates
  fail, report the exact unreachable-waypoint, joint-limit, collision, MotionGen
  status, or (only when applicable) RRT
  exhaustion gate instead of hand-authoring another asset-specific route.
- `physical.causal_rule_states[]` shows whether each rule triggered, enabled its
  effects, and released. A rule that triggered but never released usually means
  `release.maximum_driver_displacement` is too tight or `settle_s` is too short.
- Inspect `physical.causal_timing[]` whenever an object appears to animate before
  robot contact. A driver first-motion tick before its first target-contact tick,
  or an effect enable/motion tick not strictly after contact, is a generic
  causal-ordering failure; do not repair it with a per-task delay or video edit.
  If an unrelated lower/body part moves, inspect
  `undeclared_object_joints` and maximum initial displacements. Those joints
  must be held at initialization rather than assigned an inferred mechanism.
- If a spring return fights an earlier robot-owned phase, do not weaken it as a
  task-specific workaround. Its motor must remain disabled until the
  `passive_return` phase named by `control_execution`.
- If a passive joint stays at its displaced limit after release despite an
  active schedule and clear retreat, compute the inverse-dynamics generalized
  gravity load at that object state. Set the execution-data force from that
  measurement plus a disclosed generic safety factor, bounded by URDF effort;
  do not tune blindly or extend the settle window.
- If any causally triggered moving link or released spring-return link hits or
  is blocked by the gripper, arm, base, or support, do not increase its force.
  Declare `release_before_phase` no later than the earliest dependent moving
  phase. The placement solver runs
  `applications/artimo_robot_contact/solve_artimo_release_clearance.py` as the
  final per-candidate gate, copies the chosen world-frame retreat waypoint into
  `release_retreat_waypoints_world`, and copies its measured
  `minimum_release_swept_clearance_m`. Any strictly positive whole-route
  separation passes; there is no fixed 20 mm or other hidden positive margin.
  Zero or penetration fails. The solver route must replace, not follow,
  the default link-relative withdrawal and must start at the release command;
  validate and execute its exact dense joint path rather than an endpoint chord.
  Verify that release, retreat, and the nonzero safe-endpoint settle finish
  before the internal effect or passive motor is enabled. A later control-return
  phase is not a valid boundary when another triggered link moves earlier.
  Re-run placement after fixing the robot base: an old world-frame release
  waypoint must never reject a new placement. If one dense candidate has no
  release path, record that rejection and continue to the next candidate.
  If rollout collides with an internally moved link while the solver reports
  clearance, verify it projected every endpoint before `release_before_phase`
  and swept every applicable plan endpoint before the next robot-contact
  acquisition at the retreat pose. Sweeping a later manipulation after its
  reacquisition is also a generic solver failure; that motion belongs to the
  later block, while the path between contacts belongs to transit.
- If a joint reaches its target without target contact in the phase assigned to
  `robot_contact`, inspect phase ownership first. Do not lower contact gates or
  relabel the phase as an internal mechanism to preserve a passing rollout.
- For a continued grasp, verify adjacent stages share `contact_sequence` and
  invariant contact data. The next stage's sample-zero arm command must equal
  the preceding stage's final command exactly; re-solving the unchanged pose
  can produce a redundant-joint twitch even when fingers and the verified gate remain
  closed. A transit, retreat, release/reacquire, or sample-zero IK call between
  same-link stages is a harness failure, not a task-specific waypoint problem.

Convex-hull inflation is not an agent repair variable. Sparse GPU collision
uses the locked source meshes directly. Do not automatically replace them with
whole-mesh hulls or V-HACD parts: either can close real free space and reject a
valid path. If a PyBullet-only check disagrees with the source-mesh GPU result,
report a generic checker mismatch; do not write a proxy spec or
`physics_urdf` execution field and do not lower collision thresholds.

Do not spend a repair iteration re-discovering the object-side joint or effect
topology: those semantics come from `plan.json`. Repair the declared robot
contact candidate or trajectory instead.

For hidden contact, move the camera or add a crop rendered from the same frame.
Never splice another rollout. Target-contact force is not measured and is not a
repair variable; diagnose initialized overlap or a path driven through geometry
from collision pairs, penetration, clearance, and object motion instead.

## 4. Generic blocked conditions

Stop without success only when bounded search proves no reachable collision-free
contact, geometry does not expose the claimed interaction, task topology cannot
be inferred from the supplied data, or the generic execution schema lacks the
required interaction primitive. Report bounds, best residual/clearance, and the
missing primitive without mentioning a proposed asset-specific code path.
