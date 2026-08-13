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
`result.json`, but they never truncate the 65-sample trajectory, skip later
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
5. link-centered robot stance, fixed facing direction, and centerline distance;
6. continuous waypoint IK and swept-path clearance, then bounded lateral stance
   refinement only if the centered distances fail;
7. release retreat and the full passive-return swept-volume clearance;
8. target contact pair, direction, travel, and dwell for every
   robot-owned stage;
9. causal-rule state and declared internal-effect targets;
10. every plan-requested joint-motion ratio;
11. negative control and physical-video QA.

## 3. Bounded repairs

Change one execution-data class at a time in this order: contact point, contact
orientation, precontact offset, link-centered base distance, bounded lateral
base offset, intermediate waypoint, camera, then collision proxy. Arm controller
gain/force and grasp physics are fixed by the harness and are never repair
variables. Keep object orientation and facing construction fixed while
tuning distance/lateral reach unless the declared contact surface itself was
wrong. Keep search bounds and rejected candidates in debug evidence.

Use one consolidated candidate batch per variable class. The visual pass freezes
only the hard-valid contact set; it does not choose one grasp before placement.
Merge uninterrupted contacts into manipulation blocks, project every future
plan state, and jointly score base plus one visual-valid contact candidate per
block by the worst block. Select a centerline row only after this whole-task
batch, and permit one lateral batch only when all centered rows fail. When they
all fail, freeze the single closest-to-feasible whole-task centerline row and
test the declared symmetric lateral offsets only at that distance; never
multiply every failed distance by the lateral batch. Do not repeatedly invoke the planner for individual candidates,
replay the complete object motion merely to score a wrist roll, or cycle back to
a frozen class after trying later classes. If the bounded batches cannot produce
a valid path, record the exact failed clearance/IK gate as a generic gap.

Elapsed time is never a repair variable or a stopping rule. Do not shrink a
batch, keep only a candidate that happened to finish first, skip lateral
refinement, or convert a failed diagnostic row into rollout data because a
command is slow or an interactive tool window ends. Keep the same inputs and
wait, poll, resume, or rerun the byte-identical numerical command. A placement
result without an emitted `execution.json` cannot enter release planning or a
physical rollout.

Never infer wrist roll from a quaternion tuple, comparison sheet, tiled card,
or one occluded full-scene render. After fixing point, surface normal, and grasp
depth, run
`applications/artimo_robot_contact/render_artimo_grasp_orientation_candidates.py` once for the complete
roll batch. Open all four separate files for every candidate. Use the isolated
surface-normal and tangent views with cyan/magenta opposing contact links to
verify that the jaws straddle the intended feature; do not accept a roll merely
because finger shafts appear parallel to a handle or rim. Mark every candidate
visual-valid or visual-invalid with a reason, then apply the manifest through
`applications/artimo_robot_contact/apply_artimo_grasp_orientation_decisions.py`. Visual-invalid candidates
are not low-ranked candidates: the visual pass runs no IK, and invalid rolls are
absent before placement. IK, target contact geometry, and whole-arm clearance
may reject only the visual-valid set. Rank that set first by visual semantics
with unique contiguous priorities, but use priority only as a deterministic
tie-break after joint whole-task geometry. Preserve all valid candidates for the
base-and-block contact combination search. Do not use a default/template
Panda base plus five sparse driver samples as a preliminary gate. Require
two-sided target contact for an `open_then_close` grasp; do not require it for a
`maintain_width` physical push, whose actual contact/dwell is measured
during rollout. For one uninterrupted `contact_sequence`, render and probe only
the first acquisition stage, apply its roll to every member, and validate the
inherited arm branch with full placement; never probe a later member
independently from home. If no
candidate has both the required visual relationship and
valid measurements, report the bounded orientation gap instead of manually
inventing another pose.

If a visually correct roll leaves either opposing finger with a positive gap at
the final aperture, do not call it acquired and do not attach it. Reduce the
non-negative `grasp_depth_m` in execution data, rerender the immutable roll
batch, and require simultaneous PyBullet target contact from every declared
grasp link. A single allowed fingertip touching the target is not a grasp.

Dense IK continuation is harness-fixed to a 0.08-rad local trust region around
the preceding command. This prevents redundant-joint null-space wandering and
remote branch switches while preserving a finite best-local trajectory. A pose
residual at a joint limit remains diagnostic data; never remove the trust region,
restart global IK mid-trajectory, or stop the remaining rollout to reduce it.
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
  `precontact_world_m` must place every positive offset outside the object.
  Contact local +Z is outward and robot motion toward contact is -Z; never
  repair a reversed approach by making a distance negative. Panda grasptarget
  +Z runs from palm to fingertips and is converted by the harness to contact
  -Z. If the palm faces the object while fingertips point outward, repair the
  shared frame conversion, not the task quaternion.
- `precontact_offset_m` vanishes when manipulation starts, while
  `grasp_depth_m` remains. If IK succeeds but the gripper misses, read
  `target_link_gap_by_allowed_robot_link_m`; repair the declared final depth
  until the target grasp geometry is real. Do not use a large grasp depth as
  approach clearance.
- If contact is a large, short impulse followed by a miss, inspect
  `contact_acquisition` before changing offset. A grasp target approached with
  `maintain_width` or with an already-closed aperture can strike the target end
  face without ever acquiring it. Use `open_then_close` only when the open
  aperture can surround the target; tune only close duration from measured
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
- If an `open_then_close` grasp loses the target before its explicit release,
  report a shared ideal-constraint lifecycle regression. Do not tune friction,
  finger force, arm force/gain, penetration, or motion gates. Timing repair with
  `manipulation_sample_hold_s` applies only to unconstrained physical pushes.
- If an object drifts during a same-joint `hold_position` immediately after a
  robot-contact phase, inspect the serialized command schedule. A release,
  retreat, or home transit before that hold is a shared scheduler regression;
  preserve the acquired grasp through the hold instead of adding friction,
  damping, a task-specific motor, or a longer pre-release dwell.
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
  execution. It applies preceding endpoints, measures the smallest intersecting
  forbidden-link AABB, and emits four lateral plus three top-corner route
  candidates without knowing the asset. Evaluate that immutable batch once
  with `applications/artimo_robot_contact/solve_artimo_transit_clearance.py --jobs 4`. Each candidate runs
  in isolation; direct joint interpolation is attempted before bounded
  deterministic RRT, and the accepted exact joint path is serialized for
  rollout. Rank by full-robot swept clearance and path length, freeze the
  least-cost feasible route, and never replay it on retreat. If all candidates
  fail, report the exact unreachable-waypoint, joint-limit, collision, or RRT
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
  phase, run
  `applications/artimo_robot_contact/solve_artimo_release_clearance.py`, copy the chosen world-frame retreat
  waypoint into `release_retreat_waypoints_world`, and require the declared
  nonnegative `minimum_release_swept_clearance_m`. A task may declare zero when
  exact non-penetration is sufficient; no hidden positive margin may override
  it. The solver route must replace, not follow,
  the default link-relative withdrawal and must start at the release command;
  validate and execute its exact dense joint path rather than an endpoint chord.
  Verify that release, retreat, and the nonzero safe-endpoint settle finish
  before the internal effect or passive motor is enabled. A later control-return
  phase is not a valid boundary when another triggered link moves earlier.
  Re-run this solver after fixing the robot
  base: an old world-frame release waypoint must never reject a new placement.
  If rollout collides with an internally moved link while the solver reports
  clearance, verify it projected every endpoint before `release_before_phase`
  and swept every later plan endpoint at the retreat pose. Checking only robot
  drivers or passive returns is a generic solver failure.
- If a joint reaches its target without target contact in the phase assigned to
  `robot_contact`, inspect phase ownership first. Do not lower contact gates or
  relabel the phase as an internal mechanism to preserve a passing rollout.
- For a continued grasp, verify adjacent stages share `contact_sequence` and
  invariant contact data. The next stage's sample-zero arm command must equal
  the preceding stage's final command exactly; re-solving the unchanged pose
  can produce a redundant-joint twitch even when fingers and attachment remain
  closed. A transit, retreat, release/reacquire, or sample-zero IK call between
  same-link stages is a harness failure, not a task-specific waypoint problem.

If a non-target contact is an artifact of convex-hull inflation rather than real
geometry, build a collision proxy with `applications/artimo_robot_contact/build_artimo_collision_proxy.py`
under the current task's `.artimo-runs/` debug directory and pass its path as the
root `physics_urdf` field of execution data. Never edit the immutable task spec
to add a generated proxy. Never store an asset URDF, proxy spec, proxy mesh, or
report in a reusable tool/input directory; read the original URDF from the
supplied asset/data tree.
Verify the reported per-link AABB extents match the real part before trusting a
run made with it.

For collision-proxy repair, preserve source visuals, joint origins, axes, and
limits. Replace only invalid collision geometry with documented conservative
primitives. Never open a passage absent from source visual geometry.

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
