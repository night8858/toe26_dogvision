#!/usr/bin/env python3
"""Convert ONNX models to OpenVINO IR with ovc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PPOCR_ROOT = REPO_ROOT / "src" / "dogvision_vision" / "models" / "ppocr"

DEFAULT_ONNX_DIRS = {
    "det": PPOCR_ROOT / "PP-OCRv5_server_det_onnx",
    "rec": PPOCR_ROOT / "PP-OCRv5_server_rec_onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PP-OCRv5 ONNX models to OpenVINO IR."
    )
    parser.add_argument(
        "--det-onnx",
        type=Path,
        default=None,
        help="Detection ONNX file. Defaults to the only .onnx file in the det ONNX directory.",
    )
    parser.add_argument(
        "--rec-onnx",
        type=Path,
        default=None,
        help="Recognition ONNX file. Defaults to the only .onnx file in the rec ONNX directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for OpenVINO IR outputs.",
    )
    parser.add_argument(
        "--only",
        choices=("det", "rec"),
        default=None,
        help="Convert only one model.",
    )
    # parser.add_argument(
    #     "--compress-to-fp16",
    #     action="store_true",
    #     help="Ask ovc to compress model weights to FP16.",
    # )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversion commands without executing them.",
    )
    return parser.parse_args()


def selected_kinds(only: str | None) -> list[str]:
    if only:
        return [only]
    return ["det", "rec"]


def find_single_onnx(kind: str, onnx_dir: Path, dry_run: bool) -> Path:
    if not onnx_dir.is_dir():
        if dry_run:
            return onnx_dir / "inference.onnx"
        raise SystemExit(
            f"[{kind}] ONNX directory does not exist: {onnx_dir}\n"
            "Run tool/convert_paddle_to_onnx.py first, or pass --"
            f"{kind}-onnx explicitly."
        )

    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        if dry_run:
            return onnx_dir / "inference.onnx"
        raise SystemExit(f"[{kind}] no .onnx files found in: {onnx_dir}")
    if len(onnx_files) > 1:
        joined = "\n  ".join(str(path) for path in onnx_files)
        raise SystemExit(
            f"[{kind}] multiple .onnx files found; pass --{kind}-onnx explicitly:\n"
            f"  {joined}"
        )
    return onnx_files[0]


def default_output_dir(onnx_path: Path, kind: str) -> Path:
    parent_name = onnx_path.parent.name
    if parent_name.endswith("_onnx"):
        name = parent_name[: -len("_onnx")] + "_openvino"
        return onnx_path.parent.with_name(name)
    return onnx_path.parent / f"PP-OCRv5_server_{kind}_openvino"


def output_dir_for(kind: str, onnx_path: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return default_output_dir(onnx_path, kind)
    return output_root / f"PP-OCRv5_server_{kind}_openvino"


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    provided = {
        "det": args.det_onnx,
        "rec": args.rec_onnx,
    }
    inputs: dict[str, Path] = {}
    for kind in selected_kinds(args.only):
        explicit_path = provided[kind]
        if explicit_path is not None:
            onnx_path = explicit_path.resolve()
            if not args.dry_run and not onnx_path.is_file():
                raise SystemExit(f"[{kind}] ONNX file does not exist: {onnx_path}")
            if onnx_path.suffix.lower() != ".onnx":
                raise SystemExit(f"[{kind}] expected a .onnx file: {onnx_path}")
            inputs[kind] = onnx_path
        else:
            inputs[kind] = find_single_onnx(
                kind, DEFAULT_ONNX_DIRS[kind], args.dry_run
            ).resolve()
    return inputs


def resolve_ovc(dry_run: bool) -> list[str] | None:
    ovc = shutil.which("ovc")
    if ovc:
        return [ovc]

    python = shutil.which("python3") or shutil.which("python")
    if python:
        cmd = [python, "-m", "openvino.tools.ovc", "--help"]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return [python, "-m", "openvino.tools.ovc"]
        if not dry_run:
            print_openvino_help(result.stderr)
            return None

    if dry_run:
        return ["ovc"]

    print_openvino_help("")
    return None


def print_openvino_help(stderr: str) -> None:
    print(
        "OpenVINO converter was not found or could not be started.\n"
        "Load the OpenVINO environment first, for example:\n"
        "  source /home/waterking/openvino_toolkit_ubuntu24_2025.4.1.20426.82bbf0292c5_x86_64/setupvars.sh\n"
        "If Python import still fails, install the missing package reported below "
        "(this environment currently may need numpy).",
        file=sys.stderr,
    )
    if stderr.strip():
        print("\nConverter error:", file=sys.stderr)
        print(stderr.strip(), file=sys.stderr)


def command_for(
    ovc_cmd: list[str],
    onnx_path: Path,
    output_dir: Path,
    compress_to_fp16: bool,
) -> list[str]:
    cmd = [*ovc_cmd, str(onnx_path), "--output_model", str(output_dir / onnx_path.stem)]
    if compress_to_fp16:
        cmd.extend(["--compress_to_fp16", "True"])
    return cmd


def print_command(cmd: list[str]) -> None:
    print(" ".join(cmd))


def verify_openvino_ir(kind: str, output_dir: Path) -> None:
    xml_files = sorted(output_dir.glob("*.xml"))
    bin_files = sorted(output_dir.glob("*.bin"))
    if not xml_files or not bin_files:
        raise SystemExit(
            f"[{kind}] conversion finished but .xml/.bin files were not found in: "
            f"{output_dir}"
        )
    for path in xml_files + bin_files:
        print(f"[{kind}] generated: {path}")


def main() -> int:
    args = parse_args()
    inputs = resolve_inputs(args)
    ovc_cmd = resolve_ovc(args.dry_run)
    if ovc_cmd is None:
        return 2

    for kind in selected_kinds(args.only):
        onnx_path = inputs[kind]
        output_dir = output_dir_for(kind, onnx_path, args.output_root)
        cmd = command_for(ovc_cmd, onnx_path, output_dir, args.compress_to_fp16)

        print(f"[{kind}] ONNX model:     {onnx_path}")
        print(f"[{kind}] OpenVINO IR:   {output_dir}")
        print(f"[{kind}] command:")
        print_command(cmd)

        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
        verify_openvino_ir(kind, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
