# Agent workflow for asset-agnostic ArtiMo robot contact

Use one invariant program path for every object. Asset identity, paths, link
names, poses, gains, mechanisms, cameras, and search bounds are data in the task
or execution plan; they must never select Python branches or backend scripts.
The ArtiMo `plan.json` is authoritative for the object-side action graph. Do not
reconstruct that graph from URDF geometry or invent a different mechanism.

## Inputs and fixed tools

Start from a schema-v2 task JSON conforming to
`applications/artimo_robot_contact/schemas/artimo_robot_task.schema.json`. Read these documents completely:

- `applications/artimo_robot_contact/docs/acceptance-contract.md` for physical and publication gates;
- `applications/artimo_robot_contact/docs/failure-playbook.md` for bounded, asset-independent repairs.

If a later robot-contact stage must pass geometry moved by an earlier stage,
also read `applications/artimo_robot_contact/docs/obstacle-avoidance-workflow.md` completely before
proposing or scoring any transit route.

The only execution-plan format is
`applications/artimo_robot_contact/schemas/artimo_robot_execution.schema.json`.
The only simulator entry point is
`applications/artimo_robot_contact/run_artimo_physics.py`; publish its one
complete physical-plus-negative-control rollout only through
`applications/artimo_robot_contact/finalize_artimo_delivery.py`. The outer
launchers are `applications/artimo_robot_contact/run_artimo_robot_pipeline.py`
and `applications/artimo_robot_contact/run_agent_task.py`.

These generic helpers exist; use them instead of inventing a private script:

- `applications/artimo_robot_contact/inspect_artimo_contact_pose.py` renders and measures one candidate
  link-local contact point. It reports whether the point lies on the declared
  link's surface, which link is actually nearest, and the correction vector that
  would move it onto the surface. Use it before spending a rollout on a guess.
- `applications/artimo_robot_contact/render_artimo_grasp_orientation_candidates.py` rotates one fixed
  contact frame through a deterministic batch of local-surface-normal wrist
  rolls. It writes an immutable execution copy plus four separate full-resolution
  visual-only view files for every roll; it never runs IK and never creates a
  comparison sheet or tiled card. A kinematic-free parallel-jaw proxy is cyan
  and magenta in isolated views. Open every file and classify every roll before
  freezing orientation; the renderer intentionally never auto-selects one.
- `applications/artimo_robot_contact/apply_artimo_grasp_orientation_decisions.py` validates the exact four
  reviewed image hashes, one `valid`/`invalid` visual decision for every roll,
  and a unique contiguous visual priority for every visual-valid roll. It
  hard-excludes every visual-invalid roll and records their visual-priority
  order without running IK. The placement solver retains every visual-valid
  roll as a contact candidate and jointly evaluates one choice per manipulation
  block at actual candidate Panda bases. Visual priority only breaks an exact
  whole-task geometric tie. It includes
  bilateral-contact checks for `open_then_close` grasps and emits
  the only execution and chained orientation gate allowed to enter placement.
  Use it once per independent grasp acquisition in plan order. One application
  covers every consecutive stage in the same `contact_sequence`: rotate all of
  those stages together, probe only the acquisition stage, and let full
  placement validate the inherited arm branch over the complete sequence.
- `applications/artimo_robot_contact/solve_artimo_placement.py` scores bounded object/robot-base,
  wrist-orientation, and contact candidates over every sample of every declared
  robot stage. It first merges uninterrupted contact stages into manipulation
  blocks and projects `plan.json` endpoints into a kinematic shadow world before
  each block, including internal mechanisms and passive returns between robot
  contacts. It reports each block's feasible base region and accepts only their
  whole-task intersection. In `contact_facing` mode it anchors the stance on the target
  contact link's initial collision-AABB center, uses the declared contact frame
  only for the outward direction, fixes yaw toward the link center, searches
  distance on the centerline first, and tries bounded tangent offsets only if
  the centerline cannot execute the whole manipulation. It preserves prior-stage
  joint endpoints, checks the complete robot including its fixed base and
  physical support against moving forbidden links, treats the target contact
  link as collision-allowed only for `allowed_robot_contact_links`, and reports
  each allowed gripper link's gap to the target. Every other robot link and the
  support remain forbidden against the target. Run it before accepting a fixed placement;
  do not replace it with endpoint-only IK checks.
- `applications/artimo_robot_contact/solve_artimo_release_clearance.py` searches a post-release end-effector
  retreat and scores the complete robot/support against every later
  plan-declared mechanism-motion and passive-return sweep. Use it whenever
  robot contact triggers later moving geometry or a spring return follows;
  copy only its chosen pose and measured clearance into execution data. A
  solver-authored release route replaces the default link-relative retreat: it
  starts at the exact final grasp command and the planner serializes a dense
  collision-checked joint path directly to the world-frame waypoint. Never
  prepend the old approach-normal withdrawal after a door or panel has moved.
- `applications/artimo_robot_contact/propose_artimo_transit_routes.py` cheaply applies all preceding plan
  endpoints, derives the previous retreat and incoming approach endpoints from
  execution data, detects the smallest expanded forbidden-link AABB blocking
  their direct segment, and emits one immutable generic batch: four lateral
  routes plus three top-corner routes. This is an optional repair tool: skip it
  for a single robot-contact stage or whenever the direct inter-stage transit
  already passes whole-robot swept-clearance checks. Use it only after a moved
  obstacle is measured; do not hand-author coordinates from an asset-specific
  example.
- `applications/artimo_robot_contact/solve_artimo_transit_clearance.py` evaluates one immutable batch of
  geometry-derived one- or two-waypoint routes between different robot-contact
  stages in isolated candidate processes. It tries direct interpolation and
  then bounded deterministic joint-space RRT for a blocked segment, serializes
  the exact resulting joint path, ranks complete-robot swept clearance first
  and end-effector polyline length second, then emits only the frozen feasible
  execution. Use it after placement is fixed when a direct inter-stage transit
  intersects geometry moved earlier.
- `applications/artimo_robot_contact/build_artimo_collision_proxy.py` derives a physics URDF from a
  declarative proxy spec, replacing only collision geometry. Source collision
  meshes load as one convex hull per tag, which fills concavities and
  manufactures contacts; convex decomposition removes that artifact. Read the
  source URDF directly from the supplied asset/data tree. Write the proxy spec,
  derived URDF, meshes, and report only below `.artimo-runs/<task>/`, then pass
  that derived URDF as the root `physics_urdf` field in execution data. Do not
  modify the immutable task spec to add it. Read asset URDFs only from the
  supplied asset/data tree and write derived artifacts only to the current
  task's debug tree. The harness restricts an execution proxy to that task's own
  debug tree and refuses one whose joints, axes, limits, or parents differ from
  the source.
- `applications/artimo_robot_contact/review_artimo_video.py` measures the visual-QA booleans from the
  published video and the rollout traces. Never hand-write those values.
- `applications/artimo_robot_contact/artimo_plan.py` is the one parser for plan targets and ordering. The
  harness and the verifier both read it, so do not re-derive plan semantics.
- Hold object joints absent from plan drivers/passive returns/causal effects at initialization; they are not animation channels.
- `applications/artimo_robot_contact/survey_artimo_object.py` visualizes only controls already nominated as
  robot-owned. Other moving joints are reported as
  `unassigned_plan_motion_joints`, never silently coloured as internal effects;
  geometry alone cannot decide their executor. It also writes one
  `contact_link_reference__*.png` per nominated robot-contact link, with every
  other object link hidden and the selected link shown from four close-up views.
  When compact geometry protrudes from a broad link, it additionally writes
  `contact_feature_reference__*.png` crops and lists them under
  `salient_feature_references`; green marks the protrusion centre and magenta
  marks its measured bounds. Open the whole-link image and every feature crop
  before proposing any contact point. A handle,
  button, rim, lip, or latch can be a small part of a large link mesh and be
  invisible in the overview; choose the task-semantic control feature shown in
  the isolated image, never the link AABB centre or an arbitrary broad panel
  unless the declared task is direct panel pushing.

Do not call a named-asset backend, registry, profile, or stored grasp. Do not
edit harness Python for a task. If a capability is absent, report a generic
interaction-class gap; never add an asset name or calibrated constant to code.

## Cost model for search

Planning diagnostics may rank candidates before rollout, but no IK, clearance,
contact, motion-ratio, negative-control, or visual-QA boolean may stop or
truncate execution. The final selected execution always runs every `plan.json`
control in order and always exports its complete video. Diagnostics are recorded
in `result.json` for human inspection only; they are never export gates.
Candidate evaluation is numerical planning, not a sequence of visible trial
rollouts. Consolidate each variable class into one deterministic batch: one
contact point/orientation batch, one centered-distance batch, and, only if all
centered distances fail, one lateral-offset batch. Select and freeze a candidate
after each batch. Do not repeatedly reset and actuate the asset through the full
plan for individual wrist rolls or contact points. Run the complete physical
plan only once for the final frozen execution data, including its byte-identical
negative-control condition.

There is no wall-clock, elapsed-time, tool-window, or compute-time budget for a
task. A long numerical batch may be launched in the background and polled, but
runtime never authorizes narrowing or subsampling its declared candidates,
accepting a failed row, skipping the lateral batch, or moving on to rollout. If
the process is interrupted, resume it when supported or rerun the byte-identical
batch and wait for completion. Do not replace it with a smaller batch.

`solve_artimo_placement.py` writes `execution.json` only for a genuinely
feasible placement. When `placement.json` contains `execution: null`, its
`chosen` member is rejection diagnostics, not an executable fallback. Never
copy that row by hand into release-clearance input or physics execution. The
requirement to retain a complete video despite later diagnostic failures begins
only after placement has emitted a feasible `execution.json`.

The Panda is always a stiff trajectory executor, not an actuator-limit
experiment. The shared harness fixes every arm joint to 1000 N maximum motor
force, force scale 1.0, and position gain 1.0; the finger servo force is fixed
at 200 N. These fields do not exist in execution data and an agent must never
tune or diagnose them per asset. Gravity
remains enabled for the scene; visible robot sag or failure to reach a commanded
precontact pose is a shared-harness regression, not a contact/placement variable.

## Workflow

1. **Freeze inputs.** Verify the handoff lock covers the task, source/physics
   URDFs, transitive geometry, ArtiMo files, robot, schemas, and generic tools.
   Resolve the source URDF from the handed-off asset/data tree; do not search
   for alternate asset copies or treat prior generated proxies as inputs.
   Do not enumerate, grep, or read another task's `.artimo-runs/` or `outputs/`;
   the only writable/readable experiment evidence is the current task's debug
   root and eventual three-file output.
   `trajectory.jsonl` is optional and is never replayed. When present, only its
   first frame supplies non-zero initial object joints; otherwise use the URDF
   default zero state.
2. **Read the plan as the contract.** Parse every ArtiMo timeline control and,
   when present, only the trajectory's first-frame initial joint state. Preserve the plan's source phase, controlled joint,
   endpoint/extrema, return/hold semantics, and order. Ignore timestamps when
   choosing robot rollout duration. The scheduler uses one ordering tick at
   non-robot phase boundaries and never converts `t0`/`t1` into robot idle time,
   object replay, or a timed robot script.
   Run `applications/artimo_robot_contact/artimo_plan.py --plan PLAN.json` and use its exhaustive
   phase/control-index checklist, including duplicate targets and holds, as the
   rows of `control_execution[]`.
3. **Assign every control to its physical executor.** Populate
   `control_execution[]` exactly once for every `plan.json` timeline control,
   keyed by phase name and zero-based control index. Ownership is per control
   because one phase may mix a spring return with another internally actuated
   motion. Use
   `robot_contact` when continued external work by the robot is required,
   `internal_mechanism` only for a disclosed triggered transmission/actuator,
   `passive_return` only for a plan-declared `spring_return`, and `hold` only
   when no new endpoint is introduced. A phase name such as "throw", "open",
   "effect", or "settle" is not evidence of an internal actuator. The schema,
   simulator, and verifier reject missing, duplicate, or contradictory owners.
   Every `robot_contact` phase becomes a contact stage. `plan.json` is the only
   authority for contact release boundaries; never read `causal.json` to split,
   replace, or reorder robot actions. When consecutive robot-owned controls have
   no intervening plan `control_release`, they must use one uninterrupted grasp
   and one `contact_sequence`, including when the controlled joint or moving
   parent link changes. The contact link, link-local pose, finger opening,
   interaction, and allowed robot links remain identical while the robot
   continuously follows each plan-owned joint path. Do not release merely
   because a phase or driver joint changed. A different contact link is allowed
   only after an explicit plan-semantic release boundary such as a declared
   `control_release` or passive return. At a same-sequence phase boundary,
   inherit the exact final
   arm joint vector as the next stage's sample zero; never rerun IK for the
   unchanged pose. Never replace required
   robot work with a causal motor merely because the target joint is reachable
   from an earlier control.
   An `internal_mechanism` phase becomes
   `causal_rules[].source_effect_phase`; do not infer a new object-side topology
   or search for a more convenient effect link. The task execution data records
   the declared contact link/pose and robot contact link. The agent may search
   only the control's local contact point, surface normal/wrist orientation,
   approach offset, base, and waypoints.
   Classify `spring_return` as `passive_return` and `hold_position` as `hold`.
   When a `hold_position` on the same driver joint immediately follows a
   robot-contact stage, preserve that stage's acquired grasp and final arm pose
   through the hold. If it is the terminal plan phase, keep the grasp through
   the final settle frames and remove it only during simulator cleanup. Never
   release, retreat, or transit home before executing a declared hold.
   For a new `joint_position` endpoint, default to `robot_contact`. Upgrade it
   to `internal_mechanism` only after identifying the triggering control, stored
   energy/gravity/actuator that supplies the effect work, and why continued
   robot contact is unnecessary; record the source in `energy_source` and the
   evidence in `justification`. If
   those facts cannot be established from the supplied plan, task semantics,
   and visible mechanism, keep it as robot work or report ambiguous ownership.
   A `joint_position` control that repeats a joint endpoint already completed by
   an earlier control, produces no new object displacement, and is followed by
   a hold is endpoint retention, not a new manipulation. Classify it as `hold`;
   never release and reacquire merely because the repeated endpoint appears as
   another timeline control.
   When a plan drives one effect joint to a second endpoint after the control
   returns toward rest — a lid closing once a pedal is released — express that
   as the same rule's `release`, gated on measured driver return. A latched
   effect otherwise never comes back, and the plan cannot complete.
4. **Build execution data.** Write one schema-v2 execution JSON in the run/debug
   directory. Write the exhaustive `control_execution[]` ownership table before
   choosing contact poses. Every contact stage must identify its `source_phase` from
   `plan.json`, copy its driver joint and target from that phase, and express
   poses in the contacted link frame in metres and XYZW quaternion order. Every
   stage must also declare `contact_acquisition`. Choose `open_then_close` when
   the target must enter the gripper aperture before force closure (handles,
   rims, trays); choose `maintain_width` when a pre-shaped or closed gripper is
   the contact tool (buttons and direct pushes). This is an agent decision from
   geometry and task semantics, never an asset-name branch. For
   `open_then_close`, declare a larger approach opening and bounded close/settle/
   release durations. The shared schema fixes every
   `open_then_close` stage to the disclosed `explicit_ideal_feasibility`
   interaction: after physical acquisition it stays attached until the explicit
   release boundary, so frictional slip is not a search variable. Physical
   acquisition is neither proximity nor one-finger contact: every declared
   `allowed_robot_contact_links` member must simultaneously contact the target
   link before the ideal constraint may be created. The closed-aperture
   PyBullet contact probe must pass for a visually valid roll. If either finger
   retains a positive gap, reduce `grasp_depth_m`, rerender, and repeat the
   visual/physical gate; never compensate by accepting a wrong roll. For
   `maintain_width`, keep approach and manipulation widths equal, set
   close/release time to zero, and use the schema-fixed `physical_push`
   interaction. A small button may need one real finger surface rather than the
   centered Panda `grasptarget`: choose one actual robot contact link, derive
   its collision-surface point in the EEF frame from the robot URDF, and record
   that vector as `robot_tool_contact_offset_eef_m`. The harness aligns that
   tool surface—not the midpoint between the fingers—to `contact_pose_link`;
   every unselected finger/hand link remains collision-forbidden. The agent
   decides only the acquisition class and task-local tool-offset data.
   Every
   internal effect must identify its plan phase with `source_effect_phase`, have
   an explicit causal-rule id and mechanism justification, and copy that phase's
   target into its causal effect. Use only the generic
   interaction primitives and causal rules defined by the execution schema.
   Set `settle_s` long enough for any declared release to reach its endpoint;
   the default tail is one second and will cut off a slower return.
   If diagnostics require a collision proxy, keep the original task URDF
   unchanged, build the derived URDF under `.artimo-runs/<task>/`, and record its
   repository-relative path only as root `physics_urdf` in execution data.
5. **Search deterministically.** Search contact point/orientation and robot
   trajectory parameters in bounded, seeded grids. Before placement, enumerate
   the semantic control features visible in every survey
   `contact_link_reference__*.png` and `contact_feature_reference__*.png`;
   record which visible feature each candidate
   represents. Merely lying on the correct link surface is necessary but not
   sufficient. Reject a point on a door/lid/panel face when the isolated link
   image exposes a handle, button, rim, lip, or latch that is the intended
   interface. Then enumerate
   all plausible surface points and wrist orientations in one task-local
   candidate table, score them with static surface inspection plus dense
   kinematic/swept-clearance checks, select one, and freeze its link-local pose.
   Once the contact point, outward normal, and grasp depth are fixed, run
   `applications/artimo_robot_contact/render_artimo_grasp_orientation_candidates.py`. Use its default eight
   45-degree-spaced rolls unless object symmetry justifies a smaller declared
   batch. Treat independent contact acquisitions separately. Consecutive
   stages sharing one `contact_sequence` are one acquisition: render only its
   first stage, propagate the chosen roll to every member, and never re-probe a
   later member from home. Open all four separate
   `candidates/*/scene/orientation__*.png` files for every roll; never judge from
   a tiled comparison, a quaternion, or another stage's image. Judge the
   parallel-jaw closing relationship for `open_then_close`, not whether the
   wrist or finger shafts merely look parallel to a handle/rim: the cyan and
   magenta contact links must visibly straddle the intended feature on opposite
   sides. For a one-tool `maintain_width` `physical_push`, do not apply that
   bilateral-straddle test. Render the nominated real tool surface at its
   `robot_tool_contact_offset_eef_m` and judge whether that surface squarely
   covers the intended button/pad, approaches along its declared normal, and
   does not visibly sweep the surrounding housing. Because changing from a
   centered grasptarget to a nominated tool surface changes the rendered
   geometry, any earlier centered-gripper visual decisions are stale: rerender
   the full eight-roll visual-only batch before allowing any roll into IK.
   Record `valid` or `invalid` plus a concrete reason for every roll in the
   emitted decision template. Give every visual-valid roll a unique contiguous
   `visual_priority` starting at 1; rank the single best semantic and geometric
   visual choice first. A wrong visual angle is a hard exclusion even if its IK
   residual or clearance is better.
   The render report must state `visual_render_ik_was_run: false` and contain no
   numerical summaries. Run
   `applications/artimo_robot_contact/apply_artimo_grasp_orientation_decisions.py`; that step runs no IK and
   passes only the visual-valid priority list to placement. Placement retains
   every visual-valid candidate, forms the joint candidate combinations across
   independent manipulation blocks, and evaluates each bounded base against the
   complete task. It selects by the worst block first; visual priority is only a
   deterministic tie-break. Placement is coarse-to-fine. Evaluate every
   base/contact combination first at five representative manipulation states
   (start, quarter, midpoint, three-quarter, endpoint). Only those that pass
   pose residual, joint-limit, grasp-geometry, and collision gates receive a
   17-sample continuous-IK pass with joint-step checks. Rank survivors by the
   worst manipulation block; only the configured top-K (default five) may run
   adaptive dense manipulation sampling and the complete generic trajectory
   planner. The screening solver budget is tiered too: coarse defaults to four
   random restarts and 500 iterations, sparse to twelve restarts and 1000
   iterations, while dense alone inherits the execution's full IK budget.
   Placement data may lower or raise the two screening budgets, but neither is
   allowed to exceed the final execution budget. Adaptive sampling bisects
   intervals whose contacted Cartesian pose
   changes too far; it is only a precheck and never replaces final full-path IK
   and swept collision. If all centered candidates fail final dense validation,
   apply the same funnel to bounded lateral rows at the single best centerline
   distance. Never use
   the template/default Panda base plus a sparse five-point IK probe as an
   orientation gate: placement has not fixed that base, and its later dense
   path would duplicate the calculation. Bilateral contact is additionally
   required for `open_then_close`. The decision tool
   emits `execution.json` plus
   `orientation_gate.json`. The gate records every stage covered by one
   continuous sequence. For the next independent grasp acquisition, render it
   from that emitted execution and pass the previous gate with `--prior-gate`.
   Every robot-contact acquisition, including `maintain_width` pushes, must be
   covered by the chained gate. Placement must use the final gated execution
   byte-for-byte, declare its
   `orientation_gate_path`, and omit all `approach_tilt_deg`,
   `approach_spin_deg`, and `approach_roll_deg` bounds. The placement solver
   rejects missing/incomplete gates and any attempt to overwrite frozen stage
   rotations. If all visual-valid rolls fail numerical geometry, repair depth
   and rerender the complete bounded roll batch; never let a visual-invalid roll
   participate in any IK call, placement, trajectory, transit, or physical rollout.
   Do not launch a new solver command for each candidate and do not replay the
   full object motion as a visible trial. After the pose is frozen, use one
   centerline-distance batch and at most one lateral-offset batch. If those
   bounded batches fail, report the measured generic gap instead of opening new
   ad-hoc search dimensions or returning to an already frozen variable class.
   Persist only useful
   diagnostics: candidate pose, IK residual, swept-path clearance, target
   contact, and rejection reason. Do not run unrelated asset scans or
   move a calibrated value into Python.
   For a push or button control, do not keep a single canonical wrist pose:
   derive the outward surface normal from the declared link-frame contact
   patch and test at least the four approach families (along ±local X, ±local
   Y, and ±local Z as available). Contact-pose local +Z is always the outward
   normal; `grasp_depth_m` and `precontact_offset_m` are non-negative +Z
   distances, and the robot approaches in local -Z. The harness, not task data,
   applies the fixed robot-frame conversion: Panda grasptarget +Z (palm to
   fingertips) aligns with contact-frame -Z (toward the surface). The agent
   chooses the surface, magnitudes, and wrist roll, never the offset sign or a
   robot-specific 180-degree correction. Reject an
   approach that reaches the target only by
   touching an indirect-effect link; orientation is execution data, not a
   reason to change the plan or add a new stage.
   If the centered gripper overlaps geometry around a small button, first
   rerender left/right fingertip face-centroid previews for all eight rolls with
   no IK and visually gate them using the one-tool push criterion above. Then
   derive a closed support set from the selected collision face for only those
   newly visual-valid roll/tool pairs:
   its centroid, edge midpoints, and vertices in EEF coordinates; do not sample
   only one face axis or grow the set adaptively after seeing results. Run one
   bounded real-Panda numerical batch over
   `allowed_robot_contact_links` subsets and
   `robot_tool_contact_offset_eef_m`; visual-invalid rolls remain excluded from
   every IK call. Freeze the least-offset collision-free tool frame, rerender
   the final immutable eight-roll batch, and apply the ordinary orientation
   gate before placement. The immutable candidate manifest must list every
   support point before the batch starts.
   `precontact_offset_m` is transient and disappears at manipulation;
   `grasp_depth_m` remains in the final grasp. Never copy the approach distance
   into grasp depth. Accept a depth only when the placement report shows enough
   allowed gripper links within `maximum_grasp_gap_m` at every path sample and
   `target_actually_gripped: true` for every `open_then_close` stage.
   Do not infer a grasp from a short approach impact. For `open_then_close`, the
   schedule must visibly and numerically show open transit/approach, stationary
   finger closure at the first contact pose, a settled two-sided contact, then
   manipulation. For `maintain_width`, verify the maintained shape contacts the
   intended surface without an undeclared grasp-closing phase.
   If a disclosed ideal grasp loses its target before explicit release, report
   a shared constraint-lifecycle regression; do not tune force, friction,
   penetration, or arm gains. For a physical push, timing may be repaired with
   `manipulation_sample_hold_s` while preserving plan endpoints and ownership.
   Any object motion that depends on a robot-contact stage, including an
   `internal_mechanism` effect and a plan-declared `passive_return`, must never
   begin while the released gripper, arm, fixed base, or support remains in the
   moving link's swept volume. On the final preceding contact stage, declare
   `release_before_phase` no later than the earliest dependent moving phase,
   `release_retreat_waypoints_world`, and
   `minimum_release_swept_clearance_m`; obtain the waypoint with
   `applications/artimo_robot_contact/solve_artimo_release_clearance.py`. The command order is fixed:
   finish robot-owned motion, open fingers/remove any disclosed ideal grasp,
   execute the clearance retreat, hold briefly at the solved safe endpoint,
   then enable the dependent mechanism motion or passive return. The scheduler
   provides a fixed nonzero endpoint settle; an ArtiMo phase boundary is never
   treated as zero-time robot motion. Reject a candidate unless the full
   retreat path and every sampled later plan state clear the complete robot and
   support. Increasing mechanism or spring force is not a clearance repair.
   Once the contact surface is validated, write bounded placement-search data
   and run `applications/artimo_robot_contact/solve_artimo_placement.py`. Start with `contact_facing` mode.
   Before evaluating a base, merge adjacent stages sharing one uninterrupted
   `contact_sequence` into one manipulation block. Project all authoritative
   plan controls before each block into a kinematic shadow world, so a later
   tray/button/contact is evaluated with every earlier door, lid, latch,
   internal mechanism, and passive return at its planned state. Keep every
   visual-valid contact candidate for each independent block. Search the joint
   tuple `(base, block_0_contact, ..., block_N_contact)` and rank it by its worst
   block; never freeze the first block's grasp or base independently. The
   placement report records `manipulation_blocks`,
   `block_feasible_base_regions`, and `whole_task_feasible_base_region` so an
   empty fixed-base intersection is an explicit measured result.
   Ground the object once and keep its orientation fixed for a placement trial.
   Use the selected contact link's initial collision-AABB center as the stance
   anchor and the contact frame's local +Z only as its outward direction. The
   robot starts on that centerline, its yaw always faces the link center, and
   the first search varies only `contact_facing_distance_m` (plus pedestal
   height when required). Evaluate the full common feasible-base region for all
   blocks and choose its best whole-task row; only if every centered distance fails may the
   search try bounded `contact_facing_lateral_offset_m` values. Rank the complete
   failed centerline rows with the same target-grasp, IK-completeness, collision,
   and residual score used for placement; freeze the single closest-to-feasible
   centerline row, and run the one symmetric lateral batch only at that row's
   distance and other non-lateral coordinates. Do not form a Cartesian product
   of every failed distance with every lateral offset. A lateral refinement
   moves along the surface tangent but still faces the same link center. Never
   independently sweep base x/y/yaw or repeatedly rotate the
   object to repair reachability. All prior plan endpoints remain applied, and
   home-to-approach, approach, all 65 manipulation samples, and the applicable
   retreat path must clear every moving object link for the complete robot and
   support. Every stage must retain the declared target grasp geometry.

   World-frame release waypoints are valid only for the fixed base that created
   them. Placement must ignore/remove any stale
   `release_retreat_waypoints_world`, then the release-clearance solver creates
   a fresh waypoint after the base is fixed. Placement therefore validates the
   release boundary but permits its route and measured release clearance to be
   absent; the physical runner and delivery verifier remain strict and reject
   that incomplete execution. `release_before_phase` may not be
   later than the first causally dependent moving phase merely because a
   control return occurs afterward. The executable robot path ends at the safe
   waypoint, settles there, and holds there during the later object motion; do
   not invent a retreat-to-home segment.
   A later stage on a different contact releases and retreats, then transits
   directly from that retreat endpoint to the next stage's precontact path.
   Changing contacts requires a new transit, but does not imply an intermediate
   home pose. When the direct transit intersects geometry moved by an earlier
   stage, declare ordered `transit_waypoints_world` on the incoming stage. These
   poses apply only to that transit and must route around the measured obstacle;
   they are not replayed during retreat. The planner serializes the joint path
   through them and checks that exact path, and the rollout must execute the
   byte-identical serialization. Return home only after the final robot stage
   unless execution data explicitly declares a different semantic boundary.
   First run `applications/artimo_robot_contact/propose_artimo_transit_routes.py` once against the frozen
   placed execution. Confirm that its report applied every preceding endpoint
   and that the detected blocker is a declared forbidden link. Then pass its
   immutable `routes.json` to one
   `applications/artimo_robot_contact/solve_artimo_transit_clearance.py --jobs 4` invocation. Candidate
   processes must not mutate a shared execution template. Accept only the
   solver-emitted execution, whose exact serialized `transit_in` joint path is
   consumed byte-identically by physics; never invoke placement or a physical
   rollout once per route.
   Before physics, run `applications/artimo_robot_contact/visualize_artimo_scene.py` and read its frame
   diagnostics. `robot_base_outward_halfspace_m` must be positive: the fixed
   robot base must lie on the selected surface's outward/free-space side, not
   reach around from behind. `panda_eef_to_contact_inward_alignment_cosine`
   must be near +1. These are geometry diagnostics, not asset-specific axes.
   Require `target_contact_geometry_ready` and `forbidden_clearance_passed` at
   every reported sample; the tightest pair identifies whether the blocker is
   the robot base, another arm link, or the gripper.
   A placement report with `feasible: false` is never an accepted placement. If
   bounded search has no feasible row, freeze the best measured row only for the
   contract-required diagnostic rollout and label the result unsuccessful;
   generating a video does not turn that row into a passing candidate.
   When a rollout reports non-target contact, read
   `physical.contact_diagnostics[].unexpected_contact_pairs`: it names the
   offending robot and object link, the phase, and the penetration depth. Repair
   the pair it names rather than guessing.
6. **Run complete physical and negative rollouts.** Execute the same serialized robot
   commands twice. In the hidden control, disable only nominated target contact.
   Measure contacts, forces, joint travel, constraints, resets, clearances, and
   all ArtiMo-requested joint motions at every step. Dense IK is
   continuity-first: keep the current local joint branch, rank solutions by
   minimum joint-space change, constrain each 65-sample continuation solve to
   the harness-fixed 0.08-rad neighbourhood of its preceding command, treat
   Cartesian residuals only as measurements, and interpolate every pair of
   commands. The trust region constructs a smooth trajectory; it is not a
   pass/fail gate. Never discard a smooth local solution for a remote branch
   merely because of a millimetre/degree residual.
   Record `maximum_adjacent_joint_step_rad`, the minimum absolute joint-limit
   margin and its sample/joint for every stage. A smooth path that saturates a
   hard limit while pose residual grows is an infeasible local branch, not a
   successful IK result and not permission to restart global IK mid-path.
   Record commanded-versus-actual joint tracking and maximum actual joint step
   during physical and negative rollouts so collision-driven snapping is
   distinguishable from an IK command jump.
   For each causal rule, record first target contact, driver motion, latch,
   effect enable, and effect motion ticks. Driver motion cannot precede contact;
   effect enable/motion must follow it. Require every unowned object joint to
   remain within the fixed initial-state tolerance in both physical and hidden
   negative-control conditions.
   Derive gravity-loaded `passive_return` actuator capacity from inverse dynamics
   times a disclosed safety factor, bounded by URDF effort; never tune it by
   duration. Target-contact force is not measured or used as a gate.
7. **Render and inspect.** Publish the complete physical rollout. Derive the
   visual-QA booleans with `applications/artimo_robot_contact/review_artimo_video.py` at no less than the
   task's requested frame rate; it checks blank/stale frames, camera
   discontinuity, measured interpenetration depth, target contact, and requested
   motion. Preserve false values honestly but never withhold the video.
8. **Publish.** Final output always contains exactly `video.mp4`, `grasp.json`,
   and `result.json`. Verification is diagnostic evidence, not permission to
   stop or omit files. `video_exported` reports file creation independently;
   `passed` may be true only when the video exists and all measured diagnostics
   pass. Do not repeat a successful rollout for reproducibility checking.

## Invariants
- Never replay or reset object joint motion after initialization.
- Hold unowned URDF joints at initialization; fail excess displacement.
- Never replace the plan's declared source phases, targets, return semantics, or
  ordering with a URDF-derived guess.
- Never let an internal actuator execute a phase owned by `robot_contact`.
  Every robot-owned phase must have measured target contact during its own
  stage; target completion alone is insufficient.
- Never classify a phase as `internal_mechanism` from its name or desired
  motion. Require a disclosed trigger/transmission. Otherwise keep it as robot
  work or report that ownership is missing from the handoff.
- Never globally disable robot/object collisions.
- A control-triggered effect starts no earlier than `source_effect_phase` and
  only after measured displacement, target-contact dwell, and its clearance gate.
- A public video contains only causal physics: drivers cannot precede contact,
  and effects cannot precede latch/enable.
- Every `open_then_close` grasp uses one disclosed ideal fixed attachment after
  physical acquisition and retains it until explicit release. This benchmark
  tests contact-pose/IK trajectory feasibility, not frictional force closure.
  Attachment requires simultaneous target contact from every declared allowed
  robot contact link; one fingertip touching never creates it.
  `maintain_width` stages remain unconstrained physical pushes.
- Handoff/release locks require byte-identical harness code; task variation belongs only in execution data and evidence.

If a diagnostic is poor, record the measured generic limitation without adding
an asset-specialized runner. Still execute every remaining phase and publish the
complete rollout; do not relabel measured diagnostics as successful.
