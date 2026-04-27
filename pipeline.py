#!/usr/bin/env python3
"""
Motion-Aware Video Segmentation Pipeline

Automatically detects moving objects via optical flow, then tracks and
segments them with pixel-level masks using SAMURAI (motion-aware SAM2).

Usage:
    python pipeline.py --config config.yaml
    python pipeline.py --config config.yaml --input video.mp4
    python pipeline.py --config config.yaml --input frames_directory/
"""

import argparse
import json
import os
import sys

import yaml

from utils.timer import Timer
from utils.video_io import load_video
from utils.visualization import save_visualization_video, save_masks, save_flow_visualizations


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_metadata(detections, timing, output_dir):
    meta = {
        "num_objects": len(detections),
        "detections": detections,
        "timing": timing,
    }
    path = os.path.join(output_dir, "metadata.json")
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=_convert)
    print(f"  Saved metadata: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Motion-Aware Video Segmentation Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Override input video/frames path"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ config
    config = load_config(args.config)
    if args.input:
        config["video"]["input_path"] = args.input

    timer = Timer()
    input_path = config["video"]["input_path"]
    output_dir = config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------- load video
    print("Loading video...")
    frames = load_video(
        input_path,
        max_frames=config["video"].get("max_frames"),
        resize_short_edge=config["video"].get("resize_short_edge"),
    )
    print(f"Loaded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")

    if len(frames) < 2:
        print("Need at least 2 frames. Exiting.")
        sys.exit(1)

    # -------------------------------------------------- Pass 1: Optical Flow
    print("\n=== Pass 1: Computing Optical Flow ===")
    timer.start("pass1")

    from pass1_optical_flow.flow_estimator import RAFTFlowEstimator

    flow_estimator = RAFTFlowEstimator(config["pass1_optical_flow"])
    flows = flow_estimator.compute_all_flows(frames)
    timer.stop("pass1")
    print(f"Computed {len(flows)} flow maps in {timer.elapsed('pass1'):.1f}s")

    if config["output"].get("save_flow", False):
        save_flow_visualizations(flows, output_dir)

    # Free GPU memory used by RAFT
    del flow_estimator
    import torch
    torch.cuda.empty_cache()

    # ------------------------------------------------- Pass 2: Motion Analysis
    print("\n=== Pass 2: Detecting Moving Objects ===")
    timer.start("pass2")

    from pass2_motion_analysis.motion_detector import MotionDetector

    debug_dir = None
    if config["output"].get("save_detection_debug", False):
        debug_dir = os.path.join(output_dir, "debug_pass2")
    motion_detector = MotionDetector(config["pass2_motion_analysis"], debug_dir=debug_dir)
    detections = motion_detector.detect_moving_objects(flows, frames)
    timer.stop("pass2")
    print(f"Found {len(detections)} moving objects in {timer.elapsed('pass2'):.1f}s")

    for det in detections:
        print(
            f"  Object {det['object_id']}: bbox={det['initial_bbox']}, "
            f"first_frame={det['first_frame_idx']}, confidence={det['confidence']:.2f}"
        )

    if len(detections) == 0:
        print("No moving objects detected. Exiting.")
        save_metadata(detections, timer.all_elapsed(), output_dir)
        return

    # Flows no longer needed — free memory
    del flows

    # -------------------------------------------- Pass 3: SAMURAI Tracking
    print("\n=== Pass 3: Tracking with SAMURAI ===")
    timer.start("pass3")

    from pass3_tracking.samurai_tracker import SAMURAIMultiObjectTracker

    tracker = SAMURAIMultiObjectTracker(config["pass3_tracking"])
    all_masks = tracker.track_objects(frames, detections)
    timer.stop("pass3")
    print(
        f"Tracked {len(detections)} objects through {len(frames)} frames "
        f"in {timer.elapsed('pass3'):.1f}s"
    )

    # ----------------------------------------------------------- save outputs
    print("\n=== Saving Results ===")
    if config["output"].get("save_masks", True):
        save_masks(all_masks, output_dir)

    if config["output"].get("save_visualization", True):
        save_visualization_video(
            frames,
            all_masks,
            detections,
            output_dir,
            alpha=config["output"].get("visualization_alpha", 0.4),
        )

    save_metadata(detections, timer.all_elapsed(), output_dir)

    # ----------------------------------------------------------- print summary
    total = timer.elapsed("pass1") + timer.elapsed("pass2") + timer.elapsed("pass3")
    print(f"\n=== Done! Total time: {total:.1f}s for {len(frames)} frames ===")
    print(f"  Pass 1 (Flow):     {timer.elapsed('pass1'):.1f}s")
    print(f"  Pass 2 (Detect):   {timer.elapsed('pass2'):.1f}s")
    print(f"  Pass 3 (Track):    {timer.elapsed('pass3'):.1f}s")
    print(f"  Throughput:        {len(frames)/total:.1f} FPS")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
