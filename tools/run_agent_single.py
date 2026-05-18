#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ask_vlm import _extract_json_text


def find_urdf(asset_root: Path) -> Path | None:
    return next(asset_root.rglob("*.urdf"), None)


def find_glb_scene(asset_root: Path) -> Path | None:
    # Canonical textured mesh name only.
    canonical = asset_root / f"animated_textured_{asset_root.name}.glb"
    return canonical if canonical.exists() else None


def run(cmd, cwd=None):
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _apply_default_runtime_env() -> None:
    # Keep GPU-capable defaults inside the program so single runs and batch runs
    # do not need long per-command env prefixes. Explicit user env still wins.
    os.environ.setdefault("CODEX_LOOP_MOTION_RENDER_BACKEND", "blender")
    os.environ.setdefault("CODEX_BLENDER_USE_GPU", "1")
    os.environ.setdefault("CODEX_TORCH_RASTER", "1")
    os.environ.setdefault("CODEX_PYTORCH3D_VIS_RASTER", "1")
    os.environ.setdefault("CODEX_TORCH_RASTER_ALLOW_CPU", "1")


def main():
    _apply_default_runtime_env()
    parser = argparse.ArgumentParser(
        description="End-to-end single-run causal animation agent (URDF + animated mesh + action -> causal/plan/trajectory/animated GLB)"
    )
    parser.add_argument("--asset_root", required=True, help="Path to one asset folder under data/")
    parser.add_argument("--action_text", default=None, help="Single user action description (optional; falls back to asset user_prompt.txt)")
    parser.add_argument("--out_root", default="outputs", help="Root output dir; results go to <out_root>/<asset_name>")
    parser.add_argument("--vlm_model", default="gpt-5.4")
    parser.add_argument("--llm_model", default="gpt-5.4")
    parser.add_argument("--api_key", default=None, help="Single API key. Works for Gemini/OpenAI depending on --api_provider.")
    parser.add_argument("--api_provider", default="auto", choices=["auto", "openai", "gemini"])
    parser.add_argument("--api_base_url", default=None, help="Optional API base URL override")
    parser.add_argument("--resolution", type=int, nargs=2, default=[800, 600])
    parser.add_argument(
        "--use_glb_scene",
        default="auto",
        help='GLB scene for rendering: "auto" | "none" | explicit path',
    )
    parser.add_argument("--debug_motion", action="store_true")
    parser.add_argument("--enable_loop", action="store_true", help="Run coverage+motion plan loop after plan generation")
    parser.add_argument("--enable_coverage_loop", action="store_true", help="Enable coverage loop (used with --enable_loop)")
    parser.add_argument("--enable_motion_loop", action="store_true", help="Enable motion loop (used with --enable_loop)")
    parser.add_argument("--skip_coverage_loop", action="store_true", help="Skip coverage loop and reuse saved selected views when available")
    parser.add_argument("--coverage_max_iters", type=int, default=1)
    parser.add_argument("--motion_max_iters", type=int, default=3)
    parser.add_argument("--disable_loop_vlm_api", action="store_true", help="Disable GPT-based VLM in coverage/motion loops and use heuristic fallback")
    parser.add_argument(
        "--disable_numeric_verify",
        action="store_true",
        help="Compatibility no-op: ArtiMo's current motion loop uses VLM/trajectory diagnosis without the legacy numeric verifier.",
    )
    parser.add_argument(
        "--skip_plan_frames",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip_vlm", action="store_true")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--skip_plan_exec", action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument(
        "--vlm_conditioning_images",
        nargs="*",
        default=[],
        help="Optional extra object images for VLM grounding (same object).",
    )
    parser.add_argument(
        "--vlm_conditioning_masks",
        nargs="*",
        default=[],
        help="Optional extra target mask images for VLM grounding.",
    )
    parser.add_argument(
        "--vlm_conditioning_text",
        default=None,
        help="Optional extra text for VLM grounding.",
    )
    parser.add_argument(
        "--vlm_no_auto_images",
        action="store_true",
        help="If set, VLM call will only use conditioning images/masks (not generated overlay/reference images).",
    )
    parser.add_argument("--input_images", nargs="*", default=None, help="Pipeline-level alias of --vlm_conditioning_images.")
    parser.add_argument("--input_masks", nargs="*", default=None, help="Pipeline-level alias of --vlm_conditioning_masks.")
    parser.add_argument("--input_text", default=None, help="Pipeline-level extra text input for grounding; defaults to action_text.")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).absolute()
    if not asset_root.exists():
        raise SystemExit(f"Asset root not found: {asset_root}")
    if not asset_root.is_dir():
        raise SystemExit(f"Asset root is not a directory: {asset_root}")

    urdf = find_urdf(asset_root)
    if urdf is None:
        raise SystemExit(f"No URDF found under {asset_root}")

    asset_name = asset_root.name
    assets_root = asset_root.parent
    out_root = Path(args.out_root).absolute()
    asset_out = out_root / asset_name
    asset_out.mkdir(parents=True, exist_ok=True)

    user_prompt_file = asset_root / "user_prompt.txt"
    file_action_text = user_prompt_file.read_text(encoding="utf-8").strip() if user_prompt_file.exists() else ""
    effective_action_text = args.action_text if args.action_text is not None else file_action_text
    effective_input_images = list(args.input_images) if args.input_images is not None else list(args.vlm_conditioning_images)
    effective_input_masks = list(args.input_masks) if args.input_masks is not None else list(args.vlm_conditioning_masks)
    if args.input_text is not None:
        effective_input_text = args.input_text
    elif args.vlm_conditioning_text is not None:
        effective_input_text = args.vlm_conditioning_text
    else:
        effective_input_text = effective_action_text

    py = sys.executable

    # 1) Preprocess: overlays + prompts (single asset, action override)
    if not args.skip_preprocess:
        preprocess_cmd = [
            py,
            "tools/gen_overlays_and_prompts.py",
            "--assets_root",
            str(assets_root),
            "--out",
            str(out_root),
            "--asset_name",
            asset_name,
            "--resolution",
            str(args.resolution[0]),
            str(args.resolution[1]),
        ]
        if args.action_text is not None:
            preprocess_cmd.extend(["--action_text", args.action_text])
        run(preprocess_cmd)

    # 2) VLM -> causal.json/output.json
    if not args.skip_vlm:
        run(
            [
                py,
                "tools/ask_vlm.py",
                "--asset",
                asset_name,
                "--outputs_root",
                str(out_root),
                "--model",
                args.vlm_model,
                "--api_provider",
                args.api_provider,
            ]
            + (["--vlm_conditioning_images"] + effective_input_images if effective_input_images else [])
            + (["--vlm_conditioning_masks"] + effective_input_masks if effective_input_masks else [])
            + (["--vlm_conditioning_text", effective_input_text] if effective_input_text is not None else [])
            + (["--vlm_no_auto_images"] if args.vlm_no_auto_images else [])
            + (["--api_key", args.api_key] if args.api_key else [])
            + (["--api_base_url", args.api_base_url] if args.api_base_url else [])
        )
    vlm_out = asset_out / "output.json"
    if not vlm_out.exists():
        raise SystemExit(f"Missing VLM output: {vlm_out}")
    causal_json = asset_out / "causal.json"
    try:
        raw_vlm = vlm_out.read_text(encoding="utf-8")
        cleaned_vlm = _extract_json_text(raw_vlm)
        parsed = json.loads(cleaned_vlm if cleaned_vlm is not None else raw_vlm)
        causal_json.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shutil.copyfile(vlm_out, causal_json)
    print(f"Wrote {causal_json}")

    # 3) LLM plan compiler -> plan.json
    if not args.skip_llm:
        run(
            [
                py,
                "tools/ask_plan.py",
                "--asset",
                asset_name,
                "--outputs_root",
                str(out_root),
                "--asset_root",
                str(asset_root),
                "--model",
                args.llm_model,
                "--api_provider",
                args.api_provider,
            ]
            + (["--api_key", args.api_key] if args.api_key else [])
            + (["--api_base_url", args.api_base_url] if args.api_base_url else [])
        )
    plan_json = asset_out / "plan.json"
    if not plan_json.exists():
        raise SystemExit(f"Missing plan.json: {plan_json}")

    # 4) Optional automatic loop (coverage + motion patching)
    if args.enable_loop:
        loop_cmd = [
            py,
            "tools/run_agent_loop.py",
            "--asset_root",
            str(asset_root),
            "--action_text",
            effective_action_text,
            "--out_root",
            str(out_root),
            "--vlm_model",
            args.vlm_model,
            "--llm_model",
            args.llm_model,
            "--api_provider",
            args.api_provider,
            "--resolution",
            str(args.resolution[0]),
            str(args.resolution[1]),
            "--use_glb_scene",
            str(args.use_glb_scene),
            "--skip_preprocess",
            "--skip_vlm",
            "--skip_llm",
            "--coverage_max_iters",
            str(args.coverage_max_iters),
            "--motion_max_iters",
            str(args.motion_max_iters),
        ]
        if args.enable_coverage_loop:
            loop_cmd.append("--enable_coverage_loop")
        if args.enable_motion_loop:
            loop_cmd.append("--enable_motion_loop")
        if args.skip_coverage_loop:
            loop_cmd.append("--skip_coverage_loop")
        if args.debug_motion:
            loop_cmd.append("--debug_motion")
        if args.disable_loop_vlm_api:
            loop_cmd.append("--disable_loop_vlm_api")
        if effective_input_images:
            loop_cmd.extend(["--vlm_conditioning_images"] + effective_input_images)
        if effective_input_masks:
            loop_cmd.extend(["--vlm_conditioning_masks"] + effective_input_masks)
        if effective_input_text is not None:
            loop_cmd.extend(["--vlm_conditioning_text", effective_input_text])
        if args.vlm_no_auto_images:
            loop_cmd.append("--vlm_no_auto_images")
        if args.api_key:
            loop_cmd.extend(["--api_key", args.api_key])
        if args.api_base_url:
            loop_cmd.extend(["--api_base_url", args.api_base_url])
        run(loop_cmd)
        print("")
        print(f"Asset: {asset_name}")
        print(f"URDF: {urdf}")
        print(f"Output dir: {asset_out}")
        print(f"Causal JSON: {asset_out / 'causal.json'}")
        print(f"Plan JSON (looped): {asset_out / 'plan.json'}")
        print(f"Trajectory NPZ: {asset_out / 'trajectory.npz'}")
        print(f"Trajectory JSONL: {asset_out / 'trajectory.jsonl'}")
        print(f"Animated GLB: {asset_out / 'plan_animated.glb'}")
        print(f"Loop Summary: {asset_out / 'loop' / 'iterations' / 'loop_summary.json'}")
        return

    # 4) Execute plan -> trajectory + animated GLB
    if not args.skip_plan_exec:
        glb_arg = None
        if args.use_glb_scene == "auto":
            glb = find_glb_scene(asset_root)
            if glb is None:
                raise SystemExit(
                    f"Canonical textured mesh not found: "
                    f"{asset_root / f'animated_textured_{asset_root.name}.glb'}"
                )
            glb_arg = glb
        elif args.use_glb_scene.lower() == "none":
            glb_arg = None
        else:
            glb_arg = Path(args.use_glb_scene).absolute()
            if not glb_arg.exists():
                raise SystemExit(f"GLB scene not found: {glb_arg}")

        cmd = [
            py,
            "tools/run_plan.py",
            "--asset_root",
            str(asset_root),
            "--plan_json",
            str(plan_json),
            "--out",
            str(asset_out),
            "--trajectory_npz",
            str(asset_out / "trajectory.npz"),
            "--trajectory_jsonl",
            str(asset_out / "trajectory.jsonl"),
            "--resolution",
            str(args.resolution[0]),
            str(args.resolution[1]),
            "--export_animated_glb",
        ]
        if glb_arg is not None:
            cmd.extend(["--use_glb_scene", str(glb_arg)])
        if args.debug_motion:
            cmd.append("--debug_motion")
        run(cmd)

    print("")
    print(f"Asset: {asset_name}")
    print(f"URDF: {urdf}")
    print(f"Action text: {effective_action_text if effective_action_text else '[EMPTY]'}")
    print(f"Output dir: {asset_out}")
    print(f"Causal JSON: {asset_out / 'causal.json'}")
    print(f"Plan JSON: {asset_out / 'plan.json'}")
    print(f"Trajectory NPZ: {asset_out / 'trajectory.npz'}")
    print(f"Trajectory JSONL: {asset_out / 'trajectory.jsonl'}")
    print(f"Animated GLB: {asset_out / 'plan_animated.glb'}")


if __name__ == "__main__":
    main()
