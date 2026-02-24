#!/usr/bin/env python3
"""
Video stabilization preprocessing script using ffmpeg's vidstab filter.
Two-pass stabilization: detect motion -> apply transforms
"""

import subprocess
import sys
from pathlib import Path


def stabilize_video(
    input_path: str,
    output_path: str = None,
    shakiness: int = 10,
    accuracy: int = 15,
    smoothing: int = 30,
    crop: str = "black",
):
    """
    Stabilize a video using ffmpeg's vidstab filter.

    Args:
        input_path: Path to the input video file
        output_path: Path for the stabilized output (default: adds '_stabilized' suffix)
        shakiness: How shaky the video is (1-10, higher = more analysis). Default 10.
        accuracy: Accuracy of detection (1-15, higher = more accurate but slower). Default 15.
        smoothing: Number of frames for smoothing (higher = smoother but may lose fast motion). Default 30.
        crop: How to handle borders - "black" (fill with black) or "keep" (keep original pixels)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_stabilized{input_path.suffix}"
    else:
        output_path = Path(output_path)

    transforms_file = input_path.parent / f"{input_path.stem}_transforms.trf"

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Settings: shakiness={shakiness}, accuracy={accuracy}, smoothing={smoothing}, crop={crop}")
    print()

    # Pass 1: Detect motion and create transforms file
    print("Pass 1: Analyzing video motion...")
    detect_cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={transforms_file}",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(detect_cmd, capture_output=False)
    if result.returncode != 0:
        print("Error in pass 1 (motion detection)")
        sys.exit(1)

    print("Pass 1 complete.")
    print()

    # Pass 2: Apply transforms to stabilize
    print("Pass 2: Applying stabilization transforms...")

    # Determine border mode
    border_mode = "0" if crop == "black" else "1"  # 0=black, 1=keep

    transform_cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", f"vidstabtransform=input={transforms_file}:smoothing={smoothing}:crop=black:zoom=0:optzoom=1",
        "-c:a", "copy",  # Copy audio stream unchanged
        str(output_path),
        "-y"  # Overwrite output if exists
    ]

    result = subprocess.run(transform_cmd, capture_output=False)
    if result.returncode != 0:
        print("Error in pass 2 (stabilization)")
        sys.exit(1)

    print()
    print(f"Stabilization complete: {output_path}")

    # Clean up transforms file
    if transforms_file.exists():
        transforms_file.unlink()
        print("Cleaned up transforms file.")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stabilize video using ffmpeg vidstab")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output file path (default: input_stabilized.ext)")
    parser.add_argument("--shakiness", type=int, default=10,
                        help="Shakiness of input video 1-10 (default: 10)")
    parser.add_argument("--accuracy", type=int, default=15,
                        help="Detection accuracy 1-15 (default: 15)")
    parser.add_argument("--smoothing", type=int, default=30,
                        help="Smoothing frames (default: 30, higher = smoother)")
    parser.add_argument("--crop", choices=["black", "keep"], default="black",
                        help="Border handling: black or keep (default: black)")

    args = parser.parse_args()

    stabilize_video(
        args.input,
        args.output,
        shakiness=args.shakiness,
        accuracy=args.accuracy,
        smoothing=args.smoothing,
        crop=args.crop,
    )
