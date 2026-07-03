#!/usr/bin/env python3
"""Convert PaddleOCR inference models to ONNX with PaddleX paddle2onnx."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PPOCR_ROOT = REPO_ROOT / "src" / "dogvision_vision" / "models" / "ppocr"

DEFAULT_MODELS = {
    "det": PPOCR_ROOT / "PP-OCRv5_server_det_infer",
    "rec": PPOCR_ROOT / "PP-OCRv5_server_rec_infer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PP-OCRv5 Paddle inference models to ONNX."
    )
    parser.add_argument(
        "--det-model-dir",
        type=Path,
        default=DEFAULT_MODELS["det"],
        help="Paddle detection model directory.",
    )
    parser.add_argument(
        "--rec-model-dir",
        type=Path,
        default=DEFAULT_MODELS["rec"],
        help="Paddle recognition model directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for ONNX outputs.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=7,
        help="ONNX opset version passed to PaddleX paddle2onnx.",
    )
    parser.add_argument(
        "--only",
        choices=("det", "rec"),
        default=None,
        help="Convert only one model.",
    )
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


def default_output_dir(model_dir: Path) -> Path:
    name = model_dir.name
    if name.endswith("_infer"):
        name = name[: -len("_infer")] + "_onnx"
    else:
        name = name + "_onnx"
    return model_dir.with_name(name)


def output_dir_for(kind: str, model_dir: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return default_output_dir(model_dir)
    return output_root / f"PP-OCRv5_server_{kind}_onnx"


def require_model_dir(kind: str, model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(f"[{kind}] model directory does not exist: {model_dir}")

    required = ["inference.json", "inference.pdiparams"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"[{kind}] missing {', '.join(missing)} in Paddle model directory: "
            f"{model_dir}"
        )


def command_for(paddlex: str, model_dir: Path, onnx_dir: Path, opset: int) -> list[str]:
    return [
        paddlex,
        "--paddle2onnx",
        "--paddle_model_dir",
        str(model_dir),
        "--onnx_model_dir",
        str(onnx_dir),
        "--opset_version",
        str(opset),
    ]


def print_command(cmd: list[str]) -> None:
    print(" ".join(cmd))


def verify_onnx(kind: str, onnx_dir: Path) -> None:
    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(
            f"[{kind}] conversion finished but no .onnx file was found in: "
            f"{onnx_dir}"
        )
    for path in onnx_files:
        print(f"[{kind}] generated: {path}")


def main() -> int:
    args = parse_args()
    models = {
        "det": args.det_model_dir.resolve(),
        "rec": args.rec_model_dir.resolve(),
    }

    for kind in selected_kinds(args.only):
        require_model_dir(kind, models[kind])

    paddlex = shutil.which("paddlex")
    if paddlex is None:
        if not args.dry_run:
            print(
                "paddlex was not found in PATH.\n"
                "Install PaddleX, then install its paddle2onnx plugin:\n"
                "  pip install paddlex\n"
                "  paddlex --install paddle2onnx",
                file=sys.stderr,
            )
            return 2
        paddlex = "paddlex"

    for kind in selected_kinds(args.only):
        model_dir = models[kind]
        onnx_dir = output_dir_for(kind, model_dir, args.output_root)
        cmd = command_for(paddlex, model_dir, onnx_dir, args.opset)

        print(f"[{kind}] Paddle model: {model_dir}")
        print(f"[{kind}] ONNX output:  {onnx_dir}")
        print(f"[{kind}] command:")
        print_command(cmd)

        if args.dry_run:
            continue

        onnx_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
        verify_onnx(kind, onnx_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
