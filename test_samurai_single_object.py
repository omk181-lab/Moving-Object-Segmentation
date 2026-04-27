#!/usr/bin/env python3
"""
Test SAMURAI with a single object: manual bbox on frame 0, track through video.

Usage:
  # Interactive: show first frame, draw bounding box with mouse, then run SAMURAI
  python test_samurai_single_object.py --video path/to/video.mp4

  # In the window: click and drag to draw a box around the object, then press Enter.
  # Press C to cancel selection and retry.

  # Or pass bbox from command line (no GUI)
  python test_samurai_single_object.py --video path/to/video.mp4 --bbox x1 y1 x2 y2
"""

import argparse
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

SAMURAI_DIR = os.path.join(SCRIPT_DIR, "third_party", "samurai")
SAM2_DIR = os.path.join(SAMURAI_DIR, "sam2")
if os.path.isdir(SAM2_DIR):
    sys.path.insert(0, SAM2_DIR)


def select_bbox_interactive(frame: np.ndarray, window_name: str = "Select object (box, then Enter)") -> list:
    """
    Show frame and let user select a bounding box with the mouse.
    Click and drag to draw a rectangle, then press Enter to confirm (C to cancel).

    Returns [x1, y1, x2, y2] or None if cancelled.
    """
    display = frame.copy()
    h, w = display.shape[:2]
    # Resize if too large for screen (keep aspect ratio)
    max_show = 1280
    if max(h, w) > max_show:
        scale = max_show / max(h, w)
        show_w, show_h = int(w * scale), int(h * scale)
        display = cv2.resize(display, (show_w, show_h))
    else:
        show_w, show_h = w, h
        scale = 1.0

    cv2.putText(
        display, "Draw box around object, then press ENTER (C to cancel)",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    # selectROI returns (x, y, w, h) in display coords; fromCenter=False = click-drag
    roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    # roi is (x, y, w, h); empty selection gives (0,0,0,0)
    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return None

    # Back to original image coordinates if we scaled
    x1 = int(x / scale)
    y1 = int(y / scale)
    x2 = int((x + rw) / scale)
    y2 = int((y + rh) / scale)
    # Clamp to image
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x1 >= x2 or y1 >= y2:
        return None
    return [x1, y1, x2, y2]


def load_video(path: str, max_frames: int = None):
    """Load video as list of BGR frames."""
    if os.path.isdir(path):
        import glob
        exts = ("*.jpg", "*.jpeg", "*.png")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(path, ext)))
        paths.sort()
        if max_frames:
            paths = paths[:max_frames]
        frames = [cv2.imread(p) for p in paths if cv2.imread(p) is not None]
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {path}")
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
    return frames


def build_samurai_predictor(model_size: str = "base_plus", device=None):
    """Build SAMURAI video predictor (single model on GPU)."""
    from sam2.build_sam import build_sam2_video_predictor

    SAM2_CONFIGS = {
        "tiny": ("configs/samurai/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
        "small": ("configs/samurai/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
        "base_plus": ("configs/samurai/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
        "large": ("configs/samurai/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
    }
    config_file, ckpt_name = SAM2_CONFIGS[model_size]
    ckpt_path = os.path.join(SAM2_DIR, ckpt_name)
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", os.path.basename(ckpt_name))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
    )
    return predictor


def run_samurai_single_object(
    video_path: str,
    bbox: list,
    first_frame: int = 0,
    max_frames: int = None,
    model_size: str = "base_plus",
    out_dir: str = "test_samurai_out",
):
    """
    Track one object with SAMURAI from a manual bbox on the first frame.

    bbox: [x1, y1, x2, y2] in pixels (same convention as SAM2).
    """
    frames = load_video(video_path, max_frames=max_frames)
    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    bbox = [float(x) for x in bbox]
    if len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_samurai_predictor(model_size=model_size, device=device)

    temp_dir = tempfile.mkdtemp(prefix="samurai_test_")
    try:
        print(f"Saving {n_frames} frames to temp dir...")
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(temp_dir, f"{i:05d}.jpg"), f)

        inference_state = predictor.init_state(video_path=temp_dir)
        # Single object, obj_id=0, prompt on first_frame
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=first_frame,
            obj_id=0,
            box=np.array(bbox, dtype=np.float32),
        )

        masks = [None] * n_frames
        print("Propagating with SAMURAI...")
        for fi, out_obj_ids, out_mask_logits in tqdm(
            predictor.propagate_in_video(inference_state=inference_state),
            total=n_frames,
            desc="SAMURAI",
        ):
            logits = out_mask_logits[0, 0]
            mask = (logits > 0.0).cpu().numpy().astype(np.uint8) * 255
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[fi] = mask

        predictor.reset_state(inference_state)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Fill any None
    empty = np.zeros((h, w), dtype=np.uint8)
    for i in range(n_frames):
        if masks[i] is None:
            masks[i] = empty.copy()

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i, m in enumerate(masks):
        cv2.imwrite(os.path.join(masks_dir, f"{i:05d}.png"), m)

    # Overlay video
    overlay_path = os.path.join(out_dir, "overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, 30.0, (w, h))
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        vis = frame.copy()
        if mask is not None and mask.max() > 0:
            vis[mask > 127] = (vis[mask > 127] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        cv2.putText(vis, f"Frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(vis)
    writer.release()

    print(f"Done. Masks: {masks_dir}")
    print(f"Overlay video: {overlay_path}")
    return masks, overlay_path


def main():
    parser = argparse.ArgumentParser(
        description="Test SAMURAI single-object tracking. "
        "Without --bbox: show first frame and select box with mouse (drag, then Enter)."
    )
    parser.add_argument("--video", "-v", required=True, help="Video file or directory of images")
    parser.add_argument("--bbox", "-b", nargs=4, type=float, default=None,
                        metavar=("x1", "y1", "x2", "y2"),
                        help="Bounding box on first frame: x1 y1 x2 y2 (optional; if omitted, GUI selection)")
    parser.add_argument("--first-frame", type=int, default=1,
                        help="Frame index for the bbox (0-based; default: 9 = 10th frame)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (default: all)")
    parser.add_argument("--out-dir", "-o", default="test_samurai_out",
                        help="Output directory (default: test_samurai_out)")
    parser.add_argument("--model", choices=["tiny", "small", "base_plus", "large"],
                        default="base_plus", help="SAMURAI model size")
    args = parser.parse_args()

    if args.bbox is None:
        # Load first frame and let user select bbox
        frames = load_video(args.video, max_frames=max(args.first_frame + 1, 1))
        if not frames:
            raise ValueError(f"No frames in {args.video}")
        frame = frames[args.first_frame]
        print("Select the object: click and drag to draw a box, then press Enter (C to cancel).")
        bbox = select_bbox_interactive(frame)
        if bbox is None:
            print("No box selected. Exiting.")
            sys.exit(1)
        print(f"Selected bbox: [x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]")
    else:
        bbox = list(args.bbox)

    run_samurai_single_object(
        video_path=args.video,
        bbox=bbox,
        first_frame=args.first_frame,
        max_frames=args.max_frames,
        model_size=args.model,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
