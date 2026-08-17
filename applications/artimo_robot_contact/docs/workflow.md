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

## Application-owned execution defaults

Do not author a contact quaternion, Panda model fields, plan projection,
contact sequence/release boundary, forbidden-link list, passive-return gains,
GPU backend, IK budget, sparse/dense budget, or base-search matrix. Before any
schema validation or numerical work, the application materializes those fields
from the robot installation, `plan.json`, URDF collision geometry, and fixed
harness policy. A supplied value is ignored rather than treated as tuning.

For each robot-owned control, task-local input is limited to ownership, a stage
id, interaction (`explicit_ideal_feasibility` grasp or `physical_push`), the
contact link, the link-local contact-point translation, optional final finger
opening, and the visually selected wrist roll. For a physical push only,
the selected robot tool surface may also be declared. The application projects
the source phase/control, driver joint, target, uninterrupted
`contact_sequence`, release boundary and passive return directly from
`plan.json`.

The application derives roll zero from the collision-surface outward normal at
the declared point and the contacted link's longest principal tangent. It then
generates exactly `contact_roll_deg = 0/45/90/135/180`. The cached
`contact_pose_link.rotation_xyzw` and `contact_frame_source` are machine output;
never type or edit them. A fresh agent may classify the four separate renders,
but cannot redefine their common base frame. All object links except the
declared contact link are automatically forbidden. Centered Panda grasps
automatically allow exactly the two finger contact links.

Placement likewise ignores task-authored numerical search settings. It always
uses the installed GPU backend when available, a complete 5 cm contact-facing
grid, lateral coverage equal to at least the widest horizontal object-link
extent, the fixed Panda working-distance range, all visual-valid rolls, a
17-sample sparse pass, and an adaptive 65-to-129-sample dense Top-K pass.

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
  The four no-IK images place the gripper at the exact declared contact pose.
  For every roll the agent judges `angle_status` and independently records
  `contact_point_status=valid/adjust`. Except for a one-tool `physical_push`,
  `valid` means the intended feature cross-section is visibly inside the open
  gap between the cyan and magenta contact pads, with the pads on opposite
  sides, in at least one isolated view that clearly exposes the jaw-closing
  cross-section. Open and review every required view. An occluded view alone is
  not grounds for `adjust` when another isolated view clearly proves the
  relationship. Feature-pad overlap or same-side pads in the proving view, or
  the absence of any proving view, means `adjust`. Exact static
  proxy collision is not required. `adjust` means the agent must change only the
  task-local contact translation and rerender the complete five-roll batch.
  The application never moves the semantic point automatically. An agent-supplied
  `grasp_depth_m` is only the center of numerical validation, never acceptance
  evidence or authority to name excluded links.
  Placement searches the application-owned depth lattice center-out in 2.5 mm
  increments. Dense acceptance requires both finger links
  within the target-gap bound at every path sample and rejects every other
  robot/object collision. Rollout independently requires sustained real
  bilateral target-link contact, opposed contact normals, settled fingers and
  sufficient closure before enabling the disclosed contact-gated object-joint
  actuator. Its target follows the measured fractional dense-sample index along
  the current IK path through the same smoothstep used during planning, rather
  than elapsed time, linear joint interpolation, or joint-space arc length. The
  verified fingers and their nearest common rigid palm parent stop responding
  only to the target link after the gate so they cannot form a redundant solver
  loop; the remaining arm and every non-target object link stay collision-
  authoritative, and no runtime grasp constraint is created.
  Failure truncates the rollout at acquisition; no manipulation command runs.
  Visual review is a semantic plausibility check, not proof of static collision:
  the intended feature must lie in the visible open gap between both contact pads,
  rather than merely somewhere inside a large undifferentiated link. If the point
  is high, low, or on the wrong feature, mark `adjust`, repair the contact
  translation, and rerender. Dense validation and the runtime 12-tick bilateral
  gate remain the numerical and physical contact authorities.
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
  contacts. It samples every robot-contact stage at representative object-side
  fractions and records the AABB center of that stage's `driver_joint` child
  link. Samples are averaged per unique moving link and those link centers are
  then equally averaged into one whole-task placement reference. The base is
  initialized along the declared outward normal from that reference, never from
  the first contact or first moving link. It evaluates the complete declared
  distance/lateral/height/yaw/orientation grid in deterministic center-out order,
  split into consecutive bounded GPU batches; failure scores do not move the
  center or reorder the remaining cells. It preserves prior-stage
  joint endpoints, checks the complete robot including its fixed base and
  physical support against moving forbidden links, treats the target contact
  link as collision-allowed only for `allowed_robot_contact_links`, and reports
  each allowed gripper link's gap to the target. Every other robot link and the
  support remain forbidden against the target. Run it before accepting a fixed placement;
  do not replace it with endpoint-only IK checks.
- `applications/artimo_robot_contact/solve_artimo_release_clearance.py` searches a post-release end-effector
  retreat and scores the complete robot/support against plan-declared
  mechanism-motion and passive-return sweeps only until the next robot-contact
  acquisition. Later manipulation belongs to that next block, while movement
  between different contacts belongs exclusively to transit planning. Use it whenever
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
  stages. With `planning_ik_backend.name = "curobo"`, one persistent worker
  uses GPU MotionGen (graph search plus trajectory optimization) for every
  blocked joint segment and GPU collision geometry for the moved scene; it does
  not launch one CUDA model per candidate. PyBullet then verifies the complete
  returned path against exact selected links. CPU RRT exists only for explicit
  Bullet mode or `allow_bullet_fallback: true`. The solver ranks complete-robot
  swept clearance first and end-effector polyline length second, then emits only
  the frozen feasible execution. Use it after placement is fixed when a direct
  inter-stage transit intersects geometry moved earlier.
- `applications/artimo_robot_contact/build_artimo_collision_proxy.py` is a
  diagnostic-only tool. Its convex decomposition is not equivalent to the
  source mesh and must never be selected automatically for sparse, dense,
  transit, release, or rollout. Ordinary GPU collision receives the source
  meshes directly. The agent must not create a proxy spec or set `physics_urdf`
  in execution data.
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
rollouts. Contact point and visual orientation decisions are frozen before
placement. Placement uses one deterministic center-out search whose reference
represents the complete task, not one acquisition. Do not
repeatedly reset and actuate the asset through the full plan for individual
wrist rolls or contact points. Run the complete physical
plan only once for the final frozen execution data, including its byte-identical
negative-control condition.

There is no wall-clock, elapsed-time, tool-window, or compute-time budget for a
task. A long numerical batch may be launched in the background and polled, but
runtime never authorizes accepting a failed row, skipping a declared sparse
grid cell/keyframe, or moving on to rollout. If
the process is interrupted, resume it when supported or rerun the byte-identical
search and wait for completion. Do not reduce the search because it is slow.

`solve_artimo_placement.py` writes `execution.json` only for a genuinely
feasible placement. When `placement.json` contains `execution: null`, its
`chosen` member is rejection diagnostics, not an executable fallback. Never
copy that row by hand into release-clearance input or physics execution. The
requirement to retain a complete video despite later diagnostic failures begins
only after placement has emitted a feasible `execution.json`.

While placement is running it append-flushes `progress.jsonl` in its output
directory. The diagnostic stream records the declared search domain, bounded
sparse-batch construction/start/finish, dense candidate start/finish, release
acceptance, and terminal success or failure. Monitoring this file is read-only:
it may explain latency but must not alter candidate order, coverage, or gates.

In cuRobo mode, the declared contact-facing grid is ordered from its whole-task
center outward and split into consecutive multi-base GPU batches solved by one
persistent worker. Distance, lateral offset, pedestal height, yaw offset, and
visual-valid orientation retain deterministic center-out order; no failure score
moves the center or changes the remaining order. Sparse begins with five
object-side fractions per robot-contact stage. Dense validation starts
immediately when a batch contains complete sparse paths and runs every survivor
in candidate order, one candidate at a time, until one full chain passes.
Failure of the first survivor never discards the remaining survivors from that
batch. Stop immediately after dense and release both pass.
For multiple independent grasp-depth groups, dense uses coordinate search rather
than their Cartesian product: evaluate each agent seed first, search the
currently failed group's one-dimensional depth list, freeze its best value, and
continue only if a different group is then failed. A group that already passes
is not enumerated while another contact sequence remains the sole failure. If
the fully enumerated group still fails, abandon that base and evaluate the next
sparse survivor; changing another sequence's depth cannot repair it.

The Panda is always a trajectory executor, not an actuator-limit experiment.
The shared harness fixes every arm joint to 1000 N maximum motor force, force
scale 1.0, and the damped position gain 0.2; the finger servo force is fixed at
20 N so physical acquisition does not become an impact test. These fields do
not exist in execution data and an agent must never tune or diagnose them per
asset. Gravity
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
   authority for object motion and ordering; never read `causal.json` to split,
   replace, or reorder robot actions. Consecutive robot-owned controls on the
   same contact link with no intervening plan `control_release` use one
   uninterrupted grasp and one `contact_sequence`, including when the
   controlled joint or moving parent link changes. The link-local pose, finger
   opening, interaction, and allowed robot links remain identical while the
   robot follows each path. Do not release merely because a phase or driver
   joint changed. When the next robot-owned control requires a different
   physical contact link, the application automatically releases, solves a
   retreat, and reacquires before that next phase, even if the object-only plan
   has no standalone release phase. At a same-sequence phase boundary,
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
   `open_then_close`, the application derives the approach width from the
   selected final grasp width with 5 mm additional total jaw-aperture clearance
   (2.5 mm per Panda finger); it does not open every grasp to the robot maximum.
   Declare bounded close/settle/release durations. The shared schema fixes every
   `open_then_close` stage to the disclosed `explicit_ideal_feasibility`
   interaction (a legacy schema name): after runtime bilateral verification,
   it enables plan-authoritative object-joint actuation whose target follows
   measured dense-sample robot-path progress until the explicit release boundary. Frictional
   slip is not a search variable and no runtime grasp constraint is created. The
   closed-aperture agent-reviewed no-IK point gate is a semantic plausibility
   check and must pass for a visually valid roll. It does not require exact
   static proxy collision. If the feature is high, low, outside the useful jaw
   span, or visibly drives the palm into geometry, repair the task-local contact
   translation, rerender all five rolls, and repeat the visual gate; never
   compensate by accepting a wrong roll. Numerical contact confirmation is
   deferred to the dense shortlist and final rollout. For
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
   Collision meshes require no repair decision. Use the locked source collision
   meshes; never ask the agent to author a proxy spec or `physics_urdf` and
   never silently substitute a convex decomposition.
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
   Once the semantic contact point and outward normal are chosen, run
   `applications/artimo_robot_contact/render_artimo_grasp_orientation_candidates.py`. Use its default five
   45-degree-spaced robot-IK rolls (`0/45/90/135/180`). Although 0 and 180
   exchange the two fingers and are the same contact geometry, keep both because
   their Panda wrist branches, limits, and swept collisions can differ. Treat
   independent contact acquisitions separately. Consecutive
   stages sharing one `contact_sequence` are one acquisition: render only its
   first stage, propagate the chosen roll to every member, and never re-probe a
   later member from home. Open all four separate
   `candidates/*/scene/orientation__*.png` files for every roll; never judge from
   a tiled comparison, a quaternion, or another stage's image. Judge the
   parallel-jaw closing relationship for `open_then_close`, not whether the
   wrist or finger shafts merely look parallel to a handle/rim: the cyan and
   magenta contact pads must visibly straddle the intended feature cross-section
   on opposite sides in at least one isolated view that clearly exposes the
   jaw-closing cross-section. Open every required view, but an occluded view does
   not veto `valid` when another isolated view clearly proves the relationship.
   Feature-pad overlap or same-side pads in the proving view, or no proving view,
   requires `adjust`. For a one-tool `maintain_width`
   `physical_push`, do not apply that
   bilateral-straddle test. Render the nominated real tool surface at its
   `robot_tool_contact_offset_eef_m` and judge whether that surface squarely
   covers the intended button/pad, approaches along its declared normal, and
   does not visibly sweep the surrounding housing. Because changing from a
   centered grasptarget to a nominated tool surface changes the rendered
   geometry, any earlier centered-gripper visual decisions are stale: rerender
    the full five-roll visual-only batch before allowing any roll into IK.
   Record angle status first, `contact_point_status=valid/adjust` second, and final `valid`
   or `invalid` plus a concrete reason for every roll in the
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
   deterministic tie-break. Placement is sparse-to-dense. After hard-excluding
   visual-invalid orientations, construct representative object-side keyframes
   for every manipulation block. At each keyframe record the corresponding
   `driver_joint` child-link AABB center, average samples per unique moving link,
   and equally average those link centers into a whole-task reference. Initialize
   the base at the midpoint of the declared contact-facing distance range along
   the initial outward normal. Evaluate the declared placement cells in
   deterministic center-out order using consecutive bounded GPU batches.
   cuRobo performs IK, joint continuity, Panda self-collision, and conservative
   non-target environment collision on GPU. Record per-sample signed environment
   clearance and rank candidates lexicographically by complete blocks, solved
   sparse keyframes, joint margin, clearance, residual, and continuity.
   The manipulation continuity bound applies only between manipulation samples
   and across an uninterrupted `contact_sequence`. Never compare an independent
   acquisition's first grasp IK directly with home or a prior retreat and reject
   it as a manipulation discontinuity; that entry is an approach/transit path
   handled later by direct interpolation and cuRobo MotionGen (or explicit
   Bullet-mode bounded RRT). Record the raw entry step as
   a diagnostic only.
   The visual angle/offset gate is authoritative for nominal target-contact
   geometry during sparse search, so local candidates do not repeat PyBullet
   finger-gap or exact contact queries. The first sparse-complete candidate runs
   adaptive dense manipulation sampling, the complete generic trajectory
   planner, and release clearance. A dense pass whose release fails falls
   through to later local candidates; a dense manipulation failure contributes
   its failing object-side fraction to every later sparse evaluation. The first
   candidate to pass all three gates ends the search. Sparse
   defaults to twelve restarts and 1000 iterations, while dense
   alone inherits the execution's full IK budget. Placement data may lower or
   raise the sparse screening budget, but it is not allowed to exceed the final
   execution budget. Adaptive sampling bisects
   intervals whose contacted Cartesian pose
   changes too far; it is only a precheck and never replaces final full-path IK
   and swept collision. Never use
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
   full object motion as a visible trial. Use one deterministic center-out grid
   search, preserve its declared order, and batch adjacent candidates on GPU.
   Evaluate every visual-valid roll at its declared base cell. If the bounded
   grid is exhausted without a full-chain candidate, report
   the measured generic gap instead of
   opening new ad-hoc search dimensions or returning to an already frozen
   variable class.
   Persist only useful
   diagnostics: candidate pose, IK residual, swept-path clearance, target
   contact, and rejection reason. Do not run unrelated asset scans or
   move a calibrated value into Python.

   On a CUDA machine, expose cuRobo to this single placement command through
   task-local placement data; do not launch a separate process per candidate:

   ```json
   "planning_ik_backend": {
     "name": "curobo",
     "python_executable": "C:\\ProgramData\\miniforge3\\envs\\artimo-curobo\\python.exe",
     "device": "cuda:0",
     "num_seeds": 32,
     "return_seeds": 8,
     "cuda_graph": true,
     "allow_bullet_fallback": false,
     "motion_num_graph_seeds": 4,
     "motion_num_trajopt_seeds": 4,
     "motion_timeout_s": 10.0,
     "motion_max_attempts": 6
   }
   ```

   The persistent worker batches every pose in sparse and dense
   manipulation paths, returns multiple IK solutions per pose, and selects one
   whole-path joint branch under the tier's continuity bound. Dense uses the
   final fixed 0.08-rad adjacent-joint bound. Only the configured dense Top-K
   restores exact PyBullet grasp/contact-pair confirmation and the complete
   final trajectory audit; broad collision screening and clearance ranking have
   already run on GPU for every evaluated sparse cell. The selected backend is copied into
   runnable execution data, so final dense IK and blocked transit also remain on
   GPU. PyBullet remains the physical rollout engine and exact path verifier.
   If cuRobo cannot form a complete branch, Bullet refinement/RRT is used only
   when `allow_bullet_fallback` is explicitly true; the stage report records the
   exact fallback reason and must never silently present it as GPU planning. The
   CUDA default is false. Use `{"name":"bullet"}` only when CUDA is unavailable.
   For a push or button control, do not keep a single canonical wrist pose:
   derive the outward surface normal from the declared link-frame contact
   patch and test at least the four approach families (along ±local X, ±local
   Y, and ±local Z as available). Contact-pose local +Z is always the outward
   normal. `precontact_offset_m` is a non-negative +Z approach distance, while
   `grasp_depth_m` is a signed application-selected adjustment around the Panda
   centered-grasp baseline. Because Panda `grasptarget` is at the fingertips,
   the harness baseline includes a fixed 0.015 m inward useful-finger inset:
   execution value `0` therefore produces an effective `-0.015 m` pose. This is
   a robot-frame invariant and applies continuously—the effective offset is
   `-0.015 m + grasp_depth_m`, not a branch that changes only
   when the value equals zero. `maintain_width` tool/push offsets keep their
   direct signed meaning. For centered grasps the rule-based solver evaluates
   the fixed depth lattice outward from the visual seed and accepts the nearest
   value that passes bilateral target-gap and forbidden-collision checks. The robot
   approaches in local -Z. The harness, not task data,
   applies the fixed robot-frame conversion: Panda grasptarget +Z (palm to
   fingertips) aligns with contact-frame -Z (toward the surface). The agent
   chooses the surface, magnitudes, and wrist roll, never the offset sign or a
   robot-specific 180-degree correction. Reject an
   approach that reaches the target only by
   touching an indirect-effect link; orientation is execution data, not a
   reason to change the plan or add a new stage.
   If the centered gripper overlaps geometry around a small button, first
   rerender left/right fingertip face-centroid previews for all four rolls with
   no IK and visually gate them using the one-tool push criterion above. Then
   derive a closed support set from the selected collision face for only those
   newly visual-valid roll/tool pairs:
   its centroid, edge midpoints, and vertices in EEF coordinates; do not sample
   only one face axis or grow the set adaptively after seeing results. Run one
   bounded real-Panda numerical batch over
   `allowed_robot_contact_links` subsets and
   `robot_tool_contact_offset_eef_m`; visual-invalid rolls remain excluded from
   every IK call. Freeze the least-offset collision-free tool frame, rerender
   the final immutable five-roll batch, and apply the ordinary orientation
   gate before placement. The immutable candidate manifest must list every
   support point before the batch starts.
   The visual offset gate is appearance evidence, not a requirement that the
   proxy or real fingers already produce physical contact. It must show a
   convincing opposed grasp: the feature lies inside the useful finger length,
   both jaws visibly surround it, and the palm does not visibly drive into the
   object. Numerical contact and collision evidence remains deferred to dense
   validation and rollout. `precontact_offset_m` is transient and disappears at manipulation;
   `grasp_depth_m` remains in the final grasp. Never copy the approach distance
   into grasp depth. Accept a depth only when the placement report shows enough
   allowed gripper links within `maximum_grasp_gap_m` at every path sample and
   `target_actually_gripped: true` for every `open_then_close` stage.
   When the authoritative plan ends in one or more contiguous
   `hold_position` phases for the contacted driver, keep the acquired grasp at
   the final manipulation endpoint through that terminal hold. Do not append or
   collision-check an undeclared finger release, local-normal retreat, or
   return-to-home segment; planning and rollout must end on the same command.
   Do not infer a grasp from a short approach impact. For `open_then_close`, the
   schedule must visibly and numerically show open transit/approach, stationary
   an open-finger convergence dwell at the first contact pose, finger closure,
   a settled two-sided contact, then manipulation. For `maintain_width`, verify
   the maintained shape contacts the
   intended surface without an undeclared grasp-closing phase.
   If a verified stage does not move its object joint, inspect measured
   dense-sample robot-path progress, task-actuator target, measured object
   tracking error, and controller ownership; do not
   tune force, friction, penetration, or arm gains. For a physical push, timing may be repaired with
   `manipulation_sample_hold_s` while preserving plan endpoints and ownership.
   Any object motion that depends on a robot-contact stage, including an
   `internal_mechanism` effect and a plan-declared `passive_return`, must never
   begin while the released gripper, arm, fixed base, or support remains in the
   moving link's swept volume. On the final preceding contact stage, declare
   `release_before_phase` no later than the earliest dependent moving phase,
   `release_retreat_waypoints_world`, and
   `minimum_release_swept_clearance_m`; obtain the waypoint with
   `applications/artimo_robot_contact/solve_artimo_release_clearance.py`. The command order is fixed:
   finish robot-owned motion, open fingers/remove any disclosed compliant grasp,
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
   For every robot-contact stage, sample its authoritative object motion at
   `0/25/50/75/100%` and record the AABB center of the child link controlled by
   its `driver_joint`. First average all sampled positions belonging to each
   unique moving link, then equally average the resulting link centers into the
   whole-task stance reference. Use the first contact frame's local +Z only for
   the initial outward direction. Put the initial base at the midpoint of the
   declared forward-distance bounds from the whole-task reference and face that
   reference. Enumerate the declared forward/lateral/height/yaw/orientation cells
   from their centers outward in a fixed order and evaluate consecutive bounded
   GPU batches. Failure evidence cannot move the center, add a direction, or
   reorder untested cells. All prior plan endpoints remain applied, and
   home-to-approach, approach, all 65 manipulation samples, and the applicable
   retreat path must clear every moving object link for the complete robot and
   support. Sparse IK, robot self-collision, and non-target environment
   collision run as one GPU batch using the object's actual collision meshes at
   every sampled object state; whole-link AABBs are not acceptable GPU
   substitutes. Persist each complete path's GPU signed-clearance vector and
   use its minimum value in sparse ranking after complete-block and solved-frame
   counts. The nominated target link is
   excluded only from sparse
   environment collision because its opposed nominal contact was already
   frozen by the visual offset gate. Dense Top-K and final rollout restore exact
   PyBullet contact-pair validation. Every stage must retain the declared target
   grasp geometry.
   A cuRobo row with no returned solution is not by itself collision evidence.
   Report it as un-attributed GPU no-valid-solution unless the backend supplies
   a collision pair. When PyBullet finds a pose-valid, pair-clear configuration
   at the same sample, retry the GPU batch with a larger seed/return budget and
   zero feasibility-screen collision buffer; do not describe that row as a
   measured collision.

   World-frame release waypoints are valid only for the fixed base that created
   them. Placement must ignore/remove any stale
   `release_retreat_waypoints_world`, then its in-memory release-clearance gate
   creates a fresh waypoint after the base is fixed and the dense path has
   passed. Placement does not accept or emit a runnable candidate until that
   route and its measured clearance are present. A release failure returns to
   the next dense candidate rather than ending the search.
   `release_before_phase` may not be
   later than the first causally dependent moving phase merely because a
   control return occurs afterward. The executable robot path ends at the safe
   waypoint, settles there, and holds there during the later object motion; do
   not invent a retreat-to-home segment.
   A later stage on a different contact releases and retreats, then transits
   directly from that retreat endpoint to the next stage's precontact path.
   Changing contacts requires a new transit, but does not imply an intermediate
   home pose. When the direct transit intersects geometry moved by an earlier
   stage, the placement result must classify
   `prior_plan_moved_link_blocks_transit` and preserve that dense candidate as
   route-solver input; it is not an ordinary rejected placement and is not yet
   runnable. The placement command automatically runs the single bounded route
   proposal and clearance/RRT batch for the selected repair candidate. Only if
   that batch fails may the candidate be rejected. On success, declare the
   solver-emitted ordered `transit_waypoints_world` on the incoming stage. These
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
   minimum joint-space change, constrain each dense continuation solve to
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
- Every `open_then_close` grasp enables one disclosed contact-gated object-joint
  actuator only after the runtime bilateral-contact dwell gate passes, and
  retains that gate until explicit release. Its target interpolates the ArtiMo
  endpoint using monotonic measured dense-sample robot-path progress; if robot
  progress stalls, object progress stalls. The measured object-joint tracking
  error must remain within the harness tolerance. Only the verified fingers
  and their nearest common rigid palm parent stop responding to the target link
  after verification; the remaining arm and every non-target object link stay
  collision-authoritative. No runtime grasp constraint is created. This benchmark tests
  contact-pose/IK trajectory feasibility and contact-gated plan execution, not
  frictional force closure. Closure completes first; both finger links must
  simultaneously contact the target from opposed sides with settled joint
  velocity before manipulation starts.
  `maintain_width` stages remain unconstrained physical pushes.
- Handoff/release locks require byte-identical harness code; task variation belongs only in execution data and evidence.

If a diagnostic is poor, record the measured generic limitation without adding
an asset-specialized runner. Still execute every remaining phase and publish the
complete rollout; do not relabel measured diagnostics as successful.
