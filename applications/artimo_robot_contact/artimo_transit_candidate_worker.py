#!/usr/bin/env python3
"""Internal isolated worker for one transit-clearance candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import solve_artimo_transit_clearance as transit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    answer = transit._evaluate_candidate(
        request["candidate"],
        request["grounded"],
        int(request["stage_index"]),
        Path(request["simulation_urdf"]),
        Path(request["robot_urdf"]),
        {name: float(value) for name, value in request["initial"].items()},
        request["object_plan"],
        [float(value) for value in request["transit_start"]],
        [float(value) for value in request["transit_end"]],
    )
    args.response.write_text(
        json.dumps(answer, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
