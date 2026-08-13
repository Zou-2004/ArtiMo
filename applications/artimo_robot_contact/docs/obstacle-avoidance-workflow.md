# ArtiMo multi-stage moved-obstacle workflow

Read this reference completely whenever a later robot-contact stage must pass a
link moved by an earlier plan stage.

This workflow is conditional, not a mandatory phase of every task. Do not run
the route proposer for a single-stage manipulation or when the planner proves
the direct inter-stage transit clear in the already moved scene.

## Invariants

- Apply every preceding plan endpoint before measuring or planning the incoming
  transit. A door, lid, tray, or panel is an obstacle in its current moved
  state, not its initial URDF state.
- Start at the preceding released retreat endpoint and end at the incoming
  stage's outermost approach pose. Never insert home between different contacts.
- Check the complete robot, fixed base, and support. EEF-only clearance is not
  acceptance evidence.
- Keep the contact point, wrist orientation, grounded placement, plan targets,
  and task inputs frozen while repairing transit.
- Generate and score one immutable route batch. Never rewrite a shared
  execution template or run a physical rollout per candidate.

## Cost-ordered procedure

1. Confirm contact semantics and wrist roll with the survey, contact-pose
   inspector, and the four separate immutable views for every grasp-orientation
   candidate. Record `valid` or `invalid` for every candidate; a visually
   invalid roll is hard-excluded before placement, IK ranking, or transit.
2. In the kinematic shadow world, merge uninterrupted contacts into
   manipulation blocks and jointly score every visual-valid contact combination
   at every centered base. Freeze the globally selected common centerline base,
   not the first block's preferred base. Use a lateral placement only when every
   centered distance fails the full multi-block path.
3. Let the planner test the direct released-retreat-to-approach transit. If it
   clears the moved scene, keep it; do not add waypoints.
4. If the direct transit is blocked, run the cheap geometry proposer once:

   ```bash
   venv/bin/python applications/artimo_robot_contact/propose_artimo_transit_routes.py \
     --task-spec TASK.json --execution PLACED_EXECUTION.json \
     --incoming-stage STAGE_ID --out ROUTE_PROPOSAL
   ```

   The proposer derives both Cartesian endpoints from declared stage data,
   applies preceding joint endpoints, intersects their direct segment with
   expanded forbidden-link AABBs, and selects the smallest blocking volume. It
   emits four lateral-face routes and three top-corner routes at 55%, 70%, and
   85% progress. Each top route rises, crosses the nearest horizontal obstacle
   edge, rotates outside the edge, and only then descends toward contact. The
   underside is never searched because the simulator ground plane makes it
   invalid.
5. Read `proposal.json`. Verify `route_required: true`, the obstacle link is in
   the incoming stage's `forbidden_contact_links`, the reported state contains
   all preceding endpoints, and every waypoint lies outside the expanded AABB.
   Do not override the detected obstacle unless the report shows multiple
   equally intersecting declared links and visual geometry disambiguates them.
6. Evaluate the emitted batch once:

   ```bash
   venv/bin/python applications/artimo_robot_contact/solve_artimo_transit_clearance.py \
     --task-spec TASK.json --config ROUTE_PROPOSAL/routes.json \
     --out ROUTE_SOLVE --jobs 4
   ```

   Every candidate receives an immutable execution copy in an isolated
   PyBullet process. The solver checks waypoint IK, joint limits, approach and
   manipulation, the whole scheduled transit, and every prior object endpoint.
   Straight joint interpolation is tried first for each waypoint segment. If an
   elbow or wrist still crosses the obstacle, deterministic joint-space
   RRT-Connect supplies a collision-free segment with bounded adjacent steps.
   The exact resulting joint path is stored as `StagePlan.transit_in` and later
   consumed byte-identically by the rollout scheduler.
7. Accept only `ROUTE_SOLVE/execution.json` when `transit.json` reports a chosen
   feasible route. The rank is maximum whole-robot clearance, then minimum EEF
   polyline length, then stable candidate id. Re-run scene visualization on the
   chosen execution before physics.
8. Inspect the physical result's transit contact pairs and robot tracking. Zero
   penetration in the planner is insufficient if the rollout reports a new
   pair. A small commanded step with a large actual step is collision-driven
   snapping, not an IK branch jump.

## Terminal hold

If the plan immediately follows the last robot-contact stage with
`hold_position` on the same driver joint, preserve the acquired grasp and final
arm pose through the hold and rendered settle frames. Do not release, retreat,
or return home first. Simulator cleanup removes the disclosed ideal constraint
only after evidence capture.

## Bounded failure

If every emitted route fails, retain all candidate reports and name the exact
gate: unreachable waypoint, joint-limit branch, forbidden-link penetration, or
RRT exhaustion. Repair only the earlier allowed execution-data class. Do not
add an asset-specific route, disable collisions, enlarge the allowed contact
set, or repeatedly invent unbounded waypoint coordinates.
