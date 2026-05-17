#!/usr/bin/env python3
import argparse
import copy
import json
import re
from pathlib import Path

ALLOWED_EXACT = {
    "meta.fps",
    "meta.duration_s",
}
ALLOWED_PREFIXES = [
    "timeline[",
]
ALLOWED_TIMELINE_SUFFIXES = {
    ".t0",
    ".t1",
    ".axis_world",
    ".v_mps",
    ".v0_mps",
    ".tau_s",
    ".omega_radps",
    ".ramp_to_omega_radps",
    ".q_target_rad",
    ".q_target_expr",
    ".spring_k",
    ".damping_c",
    ".rest_position",
    ".min_omega_radps",
}
PROHIBITED_SUBSTRINGS = [".joint", ".joints", ".mode", ".type", "timeline[" ]

_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


def _parse_path(path: str):
    tokens = []
    i = 0
    while i < len(path):
        m = _TOKEN_RE.match(path, i)
        if not m:
            if path[i] == '.':
                i += 1
                continue
            raise ValueError(f"Invalid path syntax near: {path[i:]}")
        if m.group(1) is not None:
            tokens.append(("key", m.group(1)))
        else:
            tokens.append(("idx", int(m.group(2))))
        i = m.end()
    return tokens


def is_allowed_path(path: str) -> bool:
    if path in ALLOWED_EXACT:
        return True
    if any(s in path for s in [".joint", ".joints", ".mode", ".type"]):
        return False
    if not path.startswith("timeline["):
        return False
    # keep hard restriction to scalar parameter leaves only
    return any(path.endswith(suf) for suf in ALLOWED_TIMELINE_SUFFIXES)


def _resolve_parent(root, tokens):
    cur = root
    prefix = []
    for tkind, tval in tokens[:-1]:
        prefix.append((tkind, tval))
        if tkind == "key":
            if not isinstance(cur, dict):
                raise KeyError(f"Missing key: {tval}")
            if tval not in cur:
                raise KeyError(f"Missing key: {tval}")
            cur = cur[tval]
        else:
            if not isinstance(cur, list) or tval >= len(cur):
                raise KeyError(f"Index out of range: {tval}")
            cur = cur[tval]
    return cur, tokens[-1]


def apply_one_change(plan: dict, change: dict):
    path = str(change.get("path") or "")
    op = str(change.get("op") or "")
    if not is_allowed_path(path):
        raise ValueError(f"Disallowed patch path: {path}")
    if op not in {"replace", "scale"}:
        raise ValueError(f"Unsupported patch op: {op}")
    tokens = _parse_path(path)
    parent, leaf = _resolve_parent(plan, tokens)
    if leaf[0] == "key":
        leaf_key = leaf[1]
        if not isinstance(parent, dict):
            raise KeyError(f"Missing path leaf key: {leaf_key}")
        has_leaf = leaf_key in parent
        old = parent.get(leaf_key)
        if op == "replace":
            parent[leaf_key] = change.get("value")
        else:
            if not has_leaf:
                raise KeyError(f"Missing path leaf key: {leaf_key}")
            parent[leaf_key] = float(old) * float(change.get("value"))
    else:
        idx = leaf[1]
        if not isinstance(parent, list) or idx >= len(parent):
            raise KeyError(f"Missing path leaf index: {idx}")
        old = parent[idx]
        if op == "replace":
            parent[idx] = change.get("value")
        else:
            parent[idx] = float(old) * float(change.get("value"))


def apply_patch_to_plan(plan: dict, patch: dict) -> dict:
    if str(patch.get("patch_type") or "") != "param_only_v1":
        raise ValueError("patch_type must be param_only_v1")
    out = copy.deepcopy(plan)
    for change in patch.get("changes") or []:
        apply_one_change(out, change)
    return out


def main():
    parser = argparse.ArgumentParser(description="Apply param-only patch to plan.json")
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--patch_json", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
    patch = json.loads(Path(args.patch_json).read_text(encoding="utf-8"))
    out = apply_patch_to_plan(plan, patch)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
