#!/usr/bin/env python3
"""
End-to-end evaluation of the motion-aware segmentation pipeline on
SegTrackv2 and FBMS-Testset.

Pipeline per sequence
---------------------
1. Read all frames from ``<dataset>/JPEGImages/<seq>`` (or
   ``<dataset>/Testset/<seq>``), sorted alphabetically.
2. Encode them into a lossless H.264 .mp4 (CRF 0, yuv444p) so the existing
   ``motion_semantic_segmenter.py`` pipeline can ingest a video file
   without losing any pixel data.
3. Run ``motion_semantic_segmenter.py --segment``: this produces one
   directory ``obj_<i>/masks/`` per detected object, each containing one
   PNG per frame.
4. Take the *union* of all object masks per frame and write a single
   binary PNG to ``<res_dir>/<seq>/<basename>.png`` where ``<basename>``
   matches the input frame name (so the SegAnyMo evaluator can pair
   predictions with ground truth).
5. After all sequences complete, run SegAnyMo's ``eval_mask.py`` once per
   dataset and print the resulting J/F metrics.

Usage
-----
    python evaluate_datasets.py                # both datasets
    python evaluate_datasets.py --datasets segtrackv2
    python evaluate_datasets.py --datasets fbms --skip-pipeline   # eval only
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "dataset")
SEGANYMO_ROOT = "/home/ailab-students/big_storage_real/omkar/dynamic_object_segmentation/SegAnyMo"

# Image extensions accepted as input frames (case-insensitive).
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ---------------------------------------------------------------------------
#  Per-sequence helpers
# ---------------------------------------------------------------------------

def list_frames(seq_dir: str) -> List[str]:
    """Return absolute paths of every image directly inside ``seq_dir``,
    sorted alphabetically. Files in subdirectories (e.g. FBMS's
    ``GroundTruth`` subfolder) are ignored."""
    out: List[str] = []
    for name in sorted(os.listdir(seq_dir)):
        full = os.path.join(seq_dir, name)
        if not os.path.isfile(full):
            continue
        if name.lower().endswith(IMG_EXTS):
            out.append(full)
    return out


def build_lossless_video(frame_paths: List[str], out_video: str,
                         fps: int = 10) -> Tuple[int, int]:
    """Encode a sorted list of frame images into a lossless H.264 mp4.

    Returns (width, height) of the input frames. We re-encode the frames
    via ffmpeg with ``-crf 0 -pix_fmt yuv444p`` which is mathematically
    lossless and decodable by cv2.VideoCapture.
    """
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Cannot read {frame_paths[0]}")
    h, w = first.shape[:2]

    tmp = tempfile.mkdtemp(prefix="evds_lossless_")
    try:
        # Symlink frames to a sequential numbered pattern so ffmpeg can
        # use the simple numeric pattern matcher. We keep the original
        # extension so ffmpeg auto-detects the codec.
        ext = os.path.splitext(frame_paths[0])[1].lower()
        for i, fp in enumerate(frame_paths):
            link = os.path.join(tmp, f"f_{i:06d}{ext}")
            try:
                os.symlink(os.path.abspath(fp), link)
            except FileExistsError:
                pass
        pattern = os.path.join(tmp, f"f_%06d{ext}")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "0",
            "-pix_fmt", "yuv444p",
            out_video,
        ]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return w, h


def run_pipeline_subprocess(video_path: str, output_dir: str,
                            num_motion_frames: int = 30,
                            display_frame: int = 0,
                            threshold_percentile: float = 75.0,
                            min_area: int = 150,
                            morph_open: int = 3,
                            morph_close: int = 15,
                            temporal_decay: float = 0.85,
                            threshold_floor: float = 0.1,
                            camera_compensation: bool = True) -> int:
    """Run motion_semantic_segmenter.py on ``video_path`` and dump
    per-object SAMURAI masks to ``output_dir``.

    Defaults are more permissive than the script's CLI defaults so small
    objects (birds, frogs, worms in SegTrackv2) survive thresholding +
    morphology. ``camera_compensation`` is on by default to subtract
    global affine motion (handheld / moving-camera shots in SegTrackv2
    and FBMS).

    Returns the subprocess exit code (0 on success).
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "motion_semantic_segmenter.py"),
        "--video", video_path,
        "--output-dir", output_dir,
        "--num-frames", str(num_motion_frames),
        "--display-frame", str(display_frame),
        "--resize", "0",                          # keep original resolution
        "--split-method", "sam2",
        "--threshold-percentile", str(threshold_percentile),
        "--threshold-floor", str(threshold_floor),
        "--min-area", str(min_area),
        "--morph-open", str(morph_open),
        "--morph-close", str(morph_close),
        "--temporal-decay", str(temporal_decay),
        "--segment",
    ]
    if camera_compensation:
        cmd.append("--camera-compensation")
    log_path = os.path.join(output_dir, "pipeline.log")
    with open(log_path, "w") as log:
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return ret


def collect_per_object_masks(output_dir: str) -> List[str]:
    """Return sorted list of ``obj_*/masks`` directories produced by the
    pipeline."""
    obj_dirs = sorted(glob.glob(os.path.join(output_dir, "obj_*")))
    return [os.path.join(d, "masks") for d in obj_dirs
            if os.path.isdir(os.path.join(d, "masks"))]


def union_masks_per_frame(per_obj_dirs: List[str], n_frames: int,
                          target_hw: Tuple[int, int]) -> List[np.ndarray]:
    """For each frame index 0..n_frames-1, union all object masks at that
    index, resize to ``target_hw`` (H, W), and return as binary uint8."""
    th, tw = target_hw
    unions: List[np.ndarray] = []
    for fi in range(n_frames):
        u = np.zeros((th, tw), dtype=np.uint8)
        for d in per_obj_dirs:
            mp = os.path.join(d, f"{fi:05d}.png")
            if not os.path.isfile(mp):
                continue
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (th, tw):
                m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
            u = np.maximum(u, (m > 127).astype(np.uint8) * 255)
        unions.append(u)
    return unions


def save_predictions(unions: List[np.ndarray], frame_paths: List[str],
                     out_seq_dir: str) -> None:
    os.makedirs(out_seq_dir, exist_ok=True)
    for fp, u in zip(frame_paths, unions):
        base = os.path.splitext(os.path.basename(fp))[0]
        cv2.imwrite(os.path.join(out_seq_dir, f"{base}.png"), u)


# ---------------------------------------------------------------------------
#  Dataset processing
# ---------------------------------------------------------------------------

def process_sequence(seq_name: str, seq_img_dir: str, out_root: str,
                     num_motion_frames: int,
                     threshold_percentile: float,
                     min_area: int,
                     morph_open: int,
                     morph_close: int,
                     temporal_decay: float,
                     threshold_floor: float,
                     camera_compensation: bool) -> bool:
    """Run the full pipeline for one sequence and emit predictions to
    ``<out_root>/initial_preds/<seq_name>/``.

    Returns True on success (predictions written), False on failure (in
    which case empty masks are written so the evaluator still has files).
    """
    print(f"\n========== {seq_name} ==========")
    frames = list_frames(seq_img_dir)
    if not frames:
        print(f"  [skip] no frames in {seq_img_dir}")
        return False

    h, w = cv2.imread(frames[0]).shape[:2]
    print(f"  frames: {len(frames)} @ {w}x{h}")

    work_dir = os.path.join(out_root, "_work", seq_name)
    pred_dir = os.path.join(out_root, "initial_preds", seq_name)
    os.makedirs(work_dir, exist_ok=True)

    video_path = os.path.join(work_dir, "input.mp4")
    if not os.path.isfile(video_path):
        try:
            build_lossless_video(frames, video_path)
        except subprocess.CalledProcessError as e:
            print(f"  [error] ffmpeg failed: {e}")
            _write_empty_predictions(frames, pred_dir, (h, w))
            return False

    t0 = time.time()
    rc = run_pipeline_subprocess(
        video_path, work_dir,
        num_motion_frames=num_motion_frames,
        threshold_percentile=threshold_percentile,
        min_area=min_area,
        morph_open=morph_open,
        morph_close=morph_close,
        temporal_decay=temporal_decay,
        threshold_floor=threshold_floor,
        camera_compensation=camera_compensation,
    )
    print(f"  pipeline rc={rc} in {time.time()-t0:.1f}s")
    if rc != 0:
        print(f"  [error] pipeline returned non-zero; "
              f"see {os.path.join(work_dir, 'pipeline.log')}")
        _write_empty_predictions(frames, pred_dir, (h, w))
        return False

    obj_dirs = collect_per_object_masks(work_dir)
    # ---- Fallback retry for sequences with no detection -------------------
    # Tiny/slow objects (birdfall, parts of FBMS) accumulate so little flow
    # that the default 75th-percentile threshold on a static-camera scene
    # admits 0 active pixels. Retry once with aggressive settings: pure
    # mean+std outlier detection, no morphological opening, very small
    # min-area, and a tiny floor. This is purely additive — sequences that
    # already detected something keep their result.
    if not obj_dirs:
        print(f"  [retry] no objects detected; rerunning with aggressive "
              f"settings ...")
        # Wipe the work dir so the existing detection outputs don't confuse
        # the retry. Keep the lossless video to save time.
        for sub in os.listdir(work_dir):
            full = os.path.join(work_dir, sub)
            if full == video_path:
                continue
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
        rc = run_pipeline_subprocess(
            video_path, work_dir,
            num_motion_frames=num_motion_frames,
            threshold_percentile=50.0,
            min_area=30,
            morph_open=1,
            morph_close=9,
            temporal_decay=0.9,
            threshold_floor=0.02,
            camera_compensation=camera_compensation,
        )
        print(f"  retry rc={rc} in {time.time()-t0:.1f}s (cumulative)")
        if rc == 0:
            obj_dirs = collect_per_object_masks(work_dir)
        print(f"  detected {len(obj_dirs)} object track(s) after retry")
    else:
        print(f"  detected {len(obj_dirs)} object track(s)")
    if not obj_dirs:
        _write_empty_predictions(frames, pred_dir, (h, w))
        return True

    unions = union_masks_per_frame(obj_dirs, len(frames), (h, w))
    save_predictions(unions, frames, pred_dir)
    print(f"  -> {len(unions)} predictions saved to {pred_dir}")
    return True


def _write_empty_predictions(frames: List[str], pred_dir: str,
                             hw: Tuple[int, int]) -> None:
    h, w = hw
    os.makedirs(pred_dir, exist_ok=True)
    empty = np.zeros((h, w), dtype=np.uint8)
    for fp in frames:
        base = os.path.splitext(os.path.basename(fp))[0]
        cv2.imwrite(os.path.join(pred_dir, f"{base}.png"), empty)


def process_segtrackv2(out_root: str, sequences: List[str], **kw) -> None:
    img_root = os.path.join(DATASET_ROOT, "SegTrackv2", "JPEGImages")
    for seq in sequences:
        seq_dir = os.path.join(img_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[skip {seq}] dir not found")
            continue
        process_sequence(seq, seq_dir, out_root, **kw)


def process_fbms(out_root: str, sequences: List[str], **kw) -> None:
    img_root = os.path.join(DATASET_ROOT, "FBMS_Testset", "Testset")
    for seq in sequences:
        seq_dir = os.path.join(img_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[skip {seq}] dir not found")
            continue
        process_sequence(seq, seq_dir, out_root, **kw)


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def run_eval(dataset: str, res_dir: str) -> str:
    """Invoke SegAnyMo's eval_mask.py and capture its stdout."""
    eval_script = os.path.join(SEGANYMO_ROOT, "core", "eval", "eval_mask.py")
    util_dir = os.path.join(SEGANYMO_ROOT, "core", "utils")

    if dataset == "segtrackv2":
        cmd = [
            sys.executable, eval_script,
            "--res_dir", res_dir,
            "--eval_dir", os.path.join(DATASET_ROOT, "SegTrackv2",
                                       "GroundTruth_combined"),
            "--eval_seq_list", os.path.join(util_dir,
                                            "segtrackv2_sequences.txt"),
        ]
    elif dataset == "fbms":
        cmd = [
            sys.executable, eval_script,
            "--res_dir", res_dir,
            "--eval_dir", os.path.join(DATASET_ROOT, "FBMS_Testset",
                                       "FBMS_GT"),
            "--img_dir", os.path.join(DATASET_ROOT, "FBMS_Testset",
                                      "Testset"),
            "--eval_seq_list", os.path.join(util_dir,
                                            "fbms_sequences.txt"),
        ]
    else:
        raise ValueError(dataset)

    print(f"\n>>> Running evaluation for {dataset}")
    print(" ".join(cmd))
    # scikit-image was installed locally to .pylibs to avoid touching the
    # conda environment. The eval script needs it for the boundary metric;
    # we prepend the local site-packages directory so the subprocess
    # imports skimage from there. We do NOT do this for the pipeline
    # itself because .pylibs ships a newer numpy that conflicts with the
    # rest of the conda env.
    env = os.environ.copy()
    pylibs = os.path.join(SCRIPT_DIR, ".pylibs")
    if os.path.isdir(pylibs):
        env["PYTHONPATH"] = pylibs + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = res.stdout + ("\n[stderr]\n" + res.stderr if res.stderr else "")
    print(out)
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def _read_seq_list(path: str) -> List[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run motion-aware segmentation on SegTrackv2 and FBMS, "
                    "then evaluate with SegAnyMo's eval_mask.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+",
                        choices=["segtrackv2", "fbms"],
                        default=["segtrackv2", "fbms"])
    parser.add_argument("--out-root", default=os.path.join(SCRIPT_DIR,
                                                            "eval_outputs"))
    parser.add_argument("--num-motion-frames", type=int, default=30,
                        help="How many leading frames the motion detector "
                             "uses to build the heatmap. Lower for short "
                             "sequences.")
    parser.add_argument("--threshold-percentile", type=float, default=75.0,
                        help="Percentile threshold on the accumulated flow "
                             "magnitude (lower = more permissive).")
    parser.add_argument("--min-area", type=int, default=150,
                        help="Minimum motion-blob area in pixels.")
    parser.add_argument("--morph-open", type=int, default=3,
                        help="OPEN kernel size; small values preserve thin "
                             "fast objects (worm, bird).")
    parser.add_argument("--morph-close", type=int, default=15,
                        help="CLOSE kernel size; larger fills holes inside "
                             "articulated subjects.")
    parser.add_argument("--temporal-decay", type=float, default=0.85,
                        help="Exponential weight decay applied per-frame "
                             "around the anchor (0.85 ~ 13 effective frames).")
    parser.add_argument("--threshold-floor", type=float, default=0.1,
                        help="Minimum absolute flow magnitude (px) treated "
                             "as motion. Default 0.1; raise to 0.5 to "
                             "reject more noise at cost of small objects.")
    parser.add_argument("--no-camera-compensation", action="store_true",
                        help="Disable RANSAC global motion subtraction "
                             "(default: enabled).")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Don't re-run the pipeline; only evaluate "
                             "existing predictions in --out-root.")
    parser.add_argument("--only", nargs="*",
                        help="Restrict to a subset of sequence names.")
    args = parser.parse_args()

    util_dir = os.path.join(SEGANYMO_ROOT, "core", "utils")
    seq_lists = {
        "segtrackv2": _read_seq_list(os.path.join(util_dir,
                                                   "segtrackv2_sequences.txt")),
        "fbms": _read_seq_list(os.path.join(util_dir,
                                            "fbms_sequences.txt")),
    }

    eval_logs = {}
    for ds in args.datasets:
        seqs = seq_lists[ds]
        if args.only:
            seqs = [s for s in seqs if s in set(args.only)]
        ds_out = os.path.join(args.out_root, ds)
        os.makedirs(ds_out, exist_ok=True)

        if not args.skip_pipeline:
            print(f"\n############ Pipeline: {ds} ({len(seqs)} sequences) ############")
            t0 = time.time()
            kwargs = dict(
                num_motion_frames=args.num_motion_frames,
                threshold_percentile=args.threshold_percentile,
                min_area=args.min_area,
                morph_open=args.morph_open,
                morph_close=args.morph_close,
                temporal_decay=args.temporal_decay,
                threshold_floor=args.threshold_floor,
                camera_compensation=not args.no_camera_compensation,
            )
            if ds == "segtrackv2":
                process_segtrackv2(ds_out, seqs, **kwargs)
            else:
                process_fbms(ds_out, seqs, **kwargs)
            print(f"\n[{ds}] pipeline done in {time.time()-t0:.0f}s")

        res_dir = os.path.join(ds_out, "initial_preds")
        eval_logs[ds] = run_eval(ds, res_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for ds, log in eval_logs.items():
        print(f"\n--- {ds.upper()} ---")
        for line in log.splitlines():
            if any(tag in line for tag in (
                "J&F-Mean", "J-Mean", "F-Mean", "Global results",
            )):
                print(line)
            elif "," in line and line.replace(",", "").replace(".", "")\
                    .replace(" ", "").replace("0", "").replace("1", "")\
                    .replace("2", "").replace("3", "").replace("4", "")\
                    .replace("5", "").replace("6", "").replace("7", "")\
                    .replace("8", "").replace("9", "").replace("-", "") == "":
                print(line)


if __name__ == "__main__":
    main()
