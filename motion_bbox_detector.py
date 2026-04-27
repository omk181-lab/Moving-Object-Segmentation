#!/usr/bin/env python3
"""
Motion Region Detector — Production-Level Pipeline

Detects regions of significant motion in a video using dense optical flow
and outputs stable bounding boxes on a chosen frame (default: first),
indicating where motion occurs.

Pipeline:
  1. Dense optical flow (RAFT-Small via torchvision)
  2. Per-frame magnitude computation (optionally with camera compensation)
  3. Temporally-weighted magnitude accumulation (frames near display_frame
     are weighted exponentially higher → bbox reflects the object's position
     on that frame, not its trajectory across the whole video)
  4. Adaptive thresholding on the accumulated heatmap
  5. Morphological mask refinement
  6. Region extraction (connected components + filtering)
  7. Bbox padding and visualization

Usage:
    python motion_bbox_detector.py --video input.mp4
    python motion_bbox_detector.py --video input.mp4 --num-frames 30 --display-frame 9
    python motion_bbox_detector.py --video input.mp4 --camera-compensation

Flow Method Comparison (for reference):
  ┌────────────┬────────────────┬──────────┬───────────────────────────────────┐
  │ Method     │ Accuracy       │ Speed    │ Notes                             │
  ├────────────┼────────────────┼──────────┼───────────────────────────────────┤
  │ RAFT-Small │ ★★★★★ (best)   │ ~30ms/pr │ Sub-pixel, large displacements,   │
  │            │                │ (GPU)    │ learned features. Needs GPU.      │
  ├────────────┼────────────────┼──────────┼───────────────────────────────────┤
  │ Farneback  │ ★★★ (decent)   │ ~15ms/pr │ Fast on CPU, struggles with large │
  │            │                │ (CPU)    │ motions (>10px) and textureless.  │
  ├────────────┼────────────────┼──────────┼───────────────────────────────────┤
  │ TV-L1      │ ★★★★ (good)    │ ~80ms/pr │ Better edges than Farneback but   │
  │            │                │ (CPU)    │ very slow. No GPU acceleration.   │
  └────────────┴────────────────┴──────────┴───────────────────────────────────┘
  Recommendation: RAFT-Small for accuracy; Farneback for CPU-only speed.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.optical_flow as flow_models


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class MotionBBox:
    """A detected motion region with metadata."""
    x1: int
    y1: int
    x2: int
    y2: int
    area: int
    motion_strength: float      # mean accumulated magnitude within this region
    frame_presence: float       # what fraction of the weighted accumulation this region captures

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]


# ---------------------------------------------------------------------------
#  Core detector
# ---------------------------------------------------------------------------

class MotionBBoxDetector:
    """
    Production-level motion region detector.

    Given a video, computes optical flow, accumulates motion magnitude with
    temporal weighting centred on the display frame, then thresholds and
    extracts bounding boxes.

    Key design decisions:

    1. RAFT-Small over Farneback / TV-L1:
       RAFT handles large displacements (>10px) and textureless regions via
       learned correlation features. Farneback assumes brightness constancy
       and small motion — breaks on real-world video. TV-L1 is better but
       3-5x slower than RAFT on GPU and still inferior in accuracy.

    2. Adaptive thresholding (not fixed):
       A fixed threshold of e.g. 2.0 px/frame fails when:
       - The camera is static (all motion is object motion, magnitudes small)
       - The camera pans fast (residual after compensation is large)
       Percentile-based: "top X% of pixels are moving" auto-adapts.
       Mean + k*std: assumes Gaussian background noise, picks outliers.

    3. Camera motion compensation (RANSAC affine) — OFF by default:
       Useful when the camera is static or has independent shake/pan.
       HARMFUL when the camera tracks the subject (e.g. following a dog):
       compensation subtracts the subject's own motion, leaving only
       background residuals. Default is OFF; enable with --camera-compensation.

    4. Temporal decay centred on display_frame:
       Flow map i has weight = decay^|i - display_frame|. Frames close to
       the display frame contribute most to the heatmap. This localises the
       bbox to where the object IS on the display frame, not where it travels
       across the entire video.

    5. Accumulate raw magnitude, threshold ONCE at the end:
       Previous approach thresholded per-frame then voted. That loses
       magnitude information and fails when objects move fast (no single
       pixel stays active across enough frames). Accumulating continuous
       magnitude with temporal weighting, then thresholding once, preserves
       signal strength and naturally localises to the display frame.
    """

    def __init__(
        self,
        threshold_method: str = "percentile",
        threshold_percentile: float = 85.0,
        threshold_k: float = 2.0,
        min_object_area: int = 400,
        max_area_ratio: float = 0.6,
        max_axis_ratio: float = 0.95,
        morph_open_size: int = 5,
        morph_close_size: int = 21,
        camera_compensation: bool = False,
        temporal_decay: float = 0.6,
        border_margin: int = 5,
        bbox_padding: float = 0.12,
        batch_size: int = 8,
        threshold_floor: float = 0.5,
    ):
        """
        Args:
            threshold_method: "percentile" or "mean_std"
            threshold_percentile: percentile cutoff (used if method=percentile)
            threshold_k: number of stds above mean (used if method=mean_std)
            min_object_area: reject regions smaller than this (pixels)
            max_area_ratio: reject regions larger than this fraction of frame
            max_axis_ratio: reject bbox wider/taller than this fraction of frame
            morph_open_size: kernel size for morphological opening
            morph_close_size: kernel size for morphological closing
            camera_compensation: enable RANSAC global motion subtraction
            temporal_decay: weight decay per frame distance from display_frame
                            (0.6 means frame 5 away has weight 0.6^5 = 0.078)
            border_margin: reject detections within this many pixels of frame edge
            bbox_padding: pad each bbox by this fraction on all sides
            batch_size: RAFT batch size for frame pairs
        """
        self.threshold_method = threshold_method
        self.threshold_percentile = threshold_percentile
        self.threshold_k = threshold_k
        self.min_object_area = min_object_area
        self.max_area_ratio = max_area_ratio
        self.max_axis_ratio = max_axis_ratio
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        self.camera_compensation = camera_compensation
        self.temporal_decay = temporal_decay
        self.border_margin = border_margin
        self.bbox_padding = bbox_padding
        self.batch_size = batch_size
        self.threshold_floor = threshold_floor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # lazy init

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        video_path: str,
        num_frames: int = 40,
        display_frame: int = 0,
        resize_short_edge: Optional[int] = 480,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[MotionBBox], np.ndarray]:
        """
        Run the full pipeline on a video.

        Returns:
            (list of MotionBBox, annotated_frame as BGR numpy array)
        """
        t0 = time.time()

        # --- Step 1: Load frames ---
        print(f"[1/6] Loading up to {num_frames} frames from {video_path} ...")
        frames = self._load_frames(video_path, num_frames, resize_short_edge)
        h, w = frames[0].shape[:2]
        print(f"       Loaded {len(frames)} frames at {w}x{h}")

        if len(frames) < 3:
            print("Need at least 3 frames. Aborting.")
            return [], frames[0]

        # --- Step 2: Compute dense optical flow ---
        n_pairs = len(frames) - 1
        print(f"[2/6] Computing RAFT optical flow ({n_pairs} frame pairs) ...")
        t_flow = time.time()
        flows = self._compute_flows(frames)
        print(f"       Done in {time.time() - t_flow:.1f}s")

        # --- Step 3: Per-frame magnitude (+ optional compensation) ---
        print(f"[3/6] Computing magnitudes (camera compensation: "
              f"{'ON' if self.camera_compensation else 'OFF'}) ...")
        magnitudes = []
        for flow in flows:
            if self.camera_compensation:
                flow = self._compensate_camera(flow)
            magnitudes.append(self._compute_magnitude(flow))

        # --- Step 4: Temporally-weighted accumulation ---
        anchor = min(display_frame, len(magnitudes) - 1)
        print(f"[4/6] Accumulating with temporal decay={self.temporal_decay} "
              f"centred on frame {anchor} ...")
        heatmap = self._accumulate_magnitudes(magnitudes, anchor)

        # --- Step 5: Threshold + morphology + extract ---
        print(f"[5/6] Adaptive threshold ({self.threshold_method}) → "
              f"morphology → region extraction ...")
        mask = self._adaptive_threshold(heatmap)
        mask = self._refine_mask(mask)

        n_active = np.count_nonzero(mask)
        print(f"       Active pixels: {n_active} ({n_active / (h * w):.1%} of frame)")

        bboxes = self._extract_bboxes(mask, heatmap, h, w)
        print(f"       Found {len(bboxes)} motion region(s)")

        # --- Step 6: Visualize ---
        display = frames[min(display_frame, len(frames) - 1)].copy()
        annotated = self._visualize(display, bboxes, heatmap)

        total = time.time() - t0
        print(f"[6/6] Done in {total:.1f}s total\n")

        for i, bb in enumerate(bboxes):
            print(f"  Region {i}: bbox=[{bb.x1}, {bb.y1}, {bb.x2}, {bb.y2}]  "
                  f"size={bb.width}x{bb.height}  "
                  f"strength={bb.motion_strength:.2f}")

        if output_dir:
            self._save_outputs(output_dir, annotated, mask,
                               heatmap, bboxes, display_frame)

        return bboxes, annotated

    # ------------------------------------------------------------------ #
    #  Frame loading                                                      #
    # ------------------------------------------------------------------ #

    def _load_frames(
        self, path: str, max_frames: int, resize_short: Optional[int]
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_short is not None:
                fh, fw = frame.shape[:2]
                if min(fh, fw) != resize_short:
                    scale = resize_short / min(fh, fw)
                    frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                       interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        return frames

    # ------------------------------------------------------------------ #
    #  Dense optical flow (RAFT-Small)                                    #
    # ------------------------------------------------------------------ #

    def _init_model(self):
        if self.model is not None:
            return
        model = flow_models.raft_small(
            weights=flow_models.Raft_Small_Weights.DEFAULT
        )
        self.model = model.eval().to(self.device)
        self.transforms = flow_models.Raft_Small_Weights.DEFAULT.transforms()

    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        batch = torch.stack(tensors)
        _, _, h, w = batch.shape
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            batch = F.interpolate(batch, size=(new_h, new_w),
                                  mode="bilinear", align_corners=False)
        return batch.to(self.device)

    def _compute_flows(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        self._init_model()
        tensor = self._frames_to_tensor(frames)
        n_pairs = len(frames) - 1
        flows = []

        with torch.no_grad():
            for start in range(0, n_pairs, self.batch_size):
                end = min(start + self.batch_size, n_pairs)
                img1 = tensor[start:end]
                img2 = tensor[start + 1 : end + 1]
                img1_t, img2_t = self.transforms(img1, img2)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    preds = self.model(img1_t, img2_t)

                batch_flow = preds[-1].cpu().numpy()
                for i in range(batch_flow.shape[0]):
                    flows.append(batch_flow[i].transpose(1, 2, 0))

        del tensor
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return flows

    # ------------------------------------------------------------------ #
    #  Camera motion compensation                                         #
    # ------------------------------------------------------------------ #

    def _compensate_camera(self, flow: np.ndarray) -> np.ndarray:
        """
        Estimate global camera motion via RANSAC affine on sparse flow,
        then subtract it.

        When to use:  static camera with shake, or independent pan/tilt.
        When NOT to use:  camera tracks the subject (e.g. following a dog).
        In that case compensation removes the subject and amplifies
        background residuals.
        """
        h, w = flow.shape[:2]
        step = 16

        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yy, xx = yy.flatten(), xx.flatten()

        src = np.stack([xx, yy], axis=1).astype(np.float32)
        dst = src + flow[yy, xx]

        if len(src) < 6:
            return flow

        affine, _ = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if affine is None:
            return flow

        all_y, all_x = np.mgrid[0:h, 0:w]
        coords = np.stack([
            all_x.flatten(), all_y.flatten(), np.ones(h * w)
        ], axis=0).astype(np.float32)

        warped = affine @ coords
        camera_flow = np.stack([
            (warped[0] - coords[0]).reshape(h, w),
            (warped[1] - coords[1]).reshape(h, w),
        ], axis=-1)

        return flow - camera_flow

    # ------------------------------------------------------------------ #
    #  Magnitude + adaptive threshold                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_magnitude(flow: np.ndarray) -> np.ndarray:
        return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    def _adaptive_threshold(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Adaptive threshold on the accumulated magnitude heatmap.

        Percentile: "top X% of pixels are motion" — adapts to any video.
        Mean + k*std: statistically principled, picks outliers above the
        background noise floor.
        """
        if self.threshold_method == "percentile":
            thresh = np.percentile(magnitude, self.threshold_percentile)
            thresh = max(thresh, self.threshold_floor)
        else:
            mu = magnitude.mean()
            sigma = magnitude.std()
            thresh = mu + self.threshold_k * sigma
            thresh = max(thresh, self.threshold_floor)

        return (magnitude > thresh).astype(np.uint8) * 255

    # ------------------------------------------------------------------ #
    #  Morphological refinement                                           #
    # ------------------------------------------------------------------ #

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        OPEN removes small noise blobs and thin bridges.
        CLOSE fills holes within a single object.
        """
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_open_size, self.morph_open_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

        if self.morph_close_size > 0:
            close_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_close_size, self.morph_close_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        return mask

    # ------------------------------------------------------------------ #
    #  Temporally-weighted magnitude accumulation                         #
    # ------------------------------------------------------------------ #

    def _accumulate_magnitudes(
        self,
        magnitudes: List[np.ndarray],
        anchor_frame: int,
    ) -> np.ndarray:
        """
        Accumulate per-frame magnitudes with exponential temporal decay
        centred on anchor_frame.

        weight_i = decay ^ |i - anchor_frame|

        Why this matters:
        - A fast-moving object (e.g. a running dog) occupies DIFFERENT pixel
          locations on each frame. Uniform averaging spreads the signal along
          the trajectory; the object's position on any single frame gets diluted.
        - With temporal decay centred on the display frame, the object's
          position ON that frame receives the highest weight. Nearby frames
          (where the object is close to the same position) reinforce the signal.
          Distant frames (object far away) contribute minimally.
        - This produces a heatmap that peaks at the object's display-frame
          position rather than smearing across its entire path.

        Trade-offs:
        - Low decay (0.5): very localised, uses ~5-6 effective frames.
          Good for fast-moving objects but more susceptible to single-frame noise.
        - High decay (0.9): nearly uniform, uses all frames.
          Good for slow-moving objects or static scenes. Poor for fast movers.
        - Default 0.6: balanced, ~8 effective frames cover the display-frame
          neighbourhood without excessive smearing.
        """
        n = len(magnitudes)
        h, w = magnitudes[0].shape

        weights = np.array([
            self.temporal_decay ** abs(i - anchor_frame) for i in range(n)
        ], dtype=np.float64)
        weights /= weights.sum()

        heatmap = np.zeros((h, w), dtype=np.float64)
        for mag, wt in zip(magnitudes, weights):
            heatmap += wt * mag.astype(np.float64)

        return heatmap.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Region extraction                                                  #
    # ------------------------------------------------------------------ #

    def _extract_bboxes(
        self,
        mask: np.ndarray,
        heatmap: np.ndarray,
        h: int, w: int,
    ) -> List[MotionBBox]:
        frame_area = h * w

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x1 = stats[i, cv2.CC_STAT_LEFT]
            y1 = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            x2, y2 = x1 + bw, y1 + bh
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            if area < self.min_object_area:
                continue
            if area > frame_area * self.max_area_ratio:
                continue
            if bw > w * self.max_axis_ratio or bh > h * self.max_axis_ratio:
                continue
            # Reject very thin strips (noise from poles, edges, etc.)
            min_dim = max(30, int(min(h, w) * 0.06))
            if bw < min_dim or bh < min_dim:
                continue
            # Centroid-based border rejection
            m = self.border_margin * 4
            if m > 0 and (cx < m or cy < m or cx > w - m or cy > h - m):
                continue

            component_mask = labels == i
            strength = float(heatmap[component_mask].mean())

            # Tighten bbox using the heatmap "hot core": only keep pixels
            # within this component where magnitude > local median.
            # The coarse mask may cover a wide area (e.g. an object's full
            # trajectory), but the hottest pixels concentrate at the object's
            # actual position near the display frame.
            comp_mags = heatmap[component_mask]
            local_thresh = np.percentile(comp_mags, 60)
            hot_mask = component_mask & (heatmap >= local_thresh)
            hot_ys, hot_xs = np.where(hot_mask)

            if len(hot_ys) > 0:
                x1 = int(hot_xs.min())
                y1 = int(hot_ys.min())
                x2 = int(hot_xs.max()) + 1
                y2 = int(hot_ys.max()) + 1
                bw, bh = x2 - x1, y2 - y1

            # Pad bbox
            pad_x = int(bw * self.bbox_padding)
            pad_y = int(bh * self.bbox_padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            candidates.append(MotionBBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                area=area,
                motion_strength=strength,
                frame_presence=strength,
            ))

        candidates.sort(key=lambda b: b.motion_strength, reverse=True)
        candidates = self._nms(candidates, iou_threshold=0.3)
        return candidates

    @staticmethod
    def _nms(bboxes: List[MotionBBox], iou_threshold: float) -> List[MotionBBox]:
        """Suppress by IoU OR by containment (small box inside a larger one)."""
        if len(bboxes) <= 1:
            return bboxes
        keep = []
        suppressed = set()
        for i, a in enumerate(bboxes):
            if i in suppressed:
                continue
            keep.append(a)
            for j in range(i + 1, len(bboxes)):
                if j in suppressed:
                    continue
                b = bboxes[j]
                iou = MotionBBoxDetector._iou(a, b)
                contained = (b.x1 >= a.x1 and b.y1 >= a.y1 and
                             b.x2 <= a.x2 and b.y2 <= a.y2)
                if iou > iou_threshold or contained:
                    suppressed.add(j)
        return keep

    @staticmethod
    def _iou(a: MotionBBox, b: MotionBBox) -> float:
        xi1, yi1 = max(a.x1, b.x1), max(a.y1, b.y1)
        xi2, yi2 = min(a.x2, b.x2), min(a.y2, b.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = a.width * a.height + b.width * b.height - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  Visualization                                                      #
    # ------------------------------------------------------------------ #

    def _visualize(
        self, frame: np.ndarray, bboxes: List[MotionBBox],
        heatmap: np.ndarray,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        heatmap_norm = cv2.normalize(heatmap, None, 0, 255,
                                     cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_INFERNO)
        if heatmap_color.shape[:2] != (h, w):
            heatmap_color = cv2.resize(heatmap_color, (w, h))
        annotated = cv2.addWeighted(annotated, 0.7, heatmap_color, 0.3, 0)

        colors = [
            (0, 255, 0), (255, 100, 0), (0, 100, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]

        for i, bb in enumerate(bboxes):
            color = colors[i % len(colors)]
            cv2.rectangle(annotated, (bb.x1, bb.y1), (bb.x2, bb.y2), color, 2)

            label = f"R{i} s={bb.motion_strength:.1f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated,
                          (bb.x1, bb.y1 - th - 10),
                          (bb.x1 + tw + 6, bb.y1),
                          color, -1)
            cv2.putText(annotated, label,
                        (bb.x1 + 3, bb.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        return annotated

    def _save_outputs(
        self, output_dir: str, annotated: np.ndarray,
        mask: np.ndarray, heatmap: np.ndarray,
        bboxes: List[MotionBBox], display_frame: int,
    ):
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir, "motion_bboxes.png"), annotated)
        cv2.imwrite(os.path.join(output_dir, "motion_mask.png"), mask)

        heatmap_vis = cv2.normalize(heatmap, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, "motion_heatmap.png"), heatmap_vis)

        with open(os.path.join(output_dir, "bboxes.txt"), "w") as f:
            f.write(f"# display_frame={display_frame}\n")
            f.write(f"# format: region_id x1 y1 x2 y2 area strength\n")
            for i, bb in enumerate(bboxes):
                f.write(f"{i} {bb.x1} {bb.y1} {bb.x2} {bb.y2} "
                        f"{bb.area} {bb.motion_strength:.4f}\n")

        print(f"\n  Saved to {output_dir}/:")
        print(f"    motion_bboxes.png   — annotated frame")
        print(f"    motion_heatmap.png  — accumulated motion heatmap")
        print(f"    motion_mask.png     — binary motion mask")
        print(f"    bboxes.txt          — bbox coordinates")


# ---------------------------------------------------------------------------
#  SAMURAI segmentation
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMURAI_DIR = os.path.join(SCRIPT_DIR, "third_party", "samurai")
SAM2_DIR = os.path.join(SAMURAI_DIR, "sam2")

SAM2_CONFIGS = {
    "tiny":      ("configs/samurai/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "small":     ("configs/samurai/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/samurai/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large":     ("configs/samurai/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}


def run_samurai_segmentation(
    video_path: str,
    bbox: List[int],
    start_frame: int = 0,
    model_size: str = "base_plus",
    output_dir: str = "motion_output",
    resize_short_edge: Optional[int] = None,
) -> str:
    """
    Run SAMURAI (motion-aware SAM2) on the entire video using the given bbox
    on start_frame as the initial prompt.

    Saves:
      - Per-frame binary masks to output_dir/masks/
      - Overlay video to output_dir/segmentation.mp4

    Returns path to the overlay video.
    """
    # Add SAMURAI's sam2 to path
    if os.path.isdir(SAM2_DIR):
        sys.path.insert(0, SAM2_DIR)

    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ALL frames (not just the subset used for flow)
    print(f"\n=== SAMURAI Segmentation ===")
    print(f"  Loading full video: {video_path} ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_short_edge is not None:
            fh, fw = frame.shape[:2]
            if min(fh, fw) != resize_short_edge:
                scale = resize_short_edge / min(fh, fw)
                frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    print(f"  {n_frames} frames at {w}x{h}, fps={fps:.1f}")
    print(f"  Prompt: bbox={bbox} on frame {start_frame}")

    # Build SAMURAI predictor
    config_file, ckpt_name = SAM2_CONFIGS[model_size]
    ckpt_path = os.path.join(SAM2_DIR, "checkpoints", ckpt_name)
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", ckpt_name)

    print(f"  Loading SAMURAI ({model_size}) ...")
    predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
    )

    # SAM2 requires frames as numbered JPEGs in a directory
    temp_dir = tempfile.mkdtemp(prefix="samurai_seg_")
    try:
        print(f"  Writing frames to temp dir ...")
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(temp_dir, f"{i:05d}.jpg"), frame)

        # Init state and register the object
        inference_state = predictor.init_state(video_path=temp_dir)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=start_frame,
            obj_id=0,
            box=np.array(bbox, dtype=np.float32),
        )

        # Propagate through entire video
        masks = [None] * n_frames
        from tqdm import tqdm
        print(f"  Propagating ...")
        for fi, out_obj_ids, out_mask_logits in tqdm(
            predictor.propagate_in_video(inference_state=inference_state),
            total=n_frames,
            desc="  SAMURAI",
        ):
            logits = out_mask_logits[0, 0]
            mask = (logits > 0.0).cpu().numpy().astype(np.uint8) * 255
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[fi] = mask

        predictor.reset_state(inference_state)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Fill any None frames with empty masks
    empty = np.zeros((h, w), dtype=np.uint8)
    for i in range(n_frames):
        if masks[i] is None:
            masks[i] = empty.copy()

    # Save masks
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i, m in enumerate(masks):
        cv2.imwrite(os.path.join(masks_dir, f"{i:05d}.png"), m)
    print(f"  Saved {n_frames} masks to {masks_dir}/")

    # Save overlay video
    overlay_path = os.path.join(output_dir, "segmentation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        vis = frame.copy()
        if mask is not None and mask.max() > 0:
            green_overlay = np.zeros_like(vis)
            green_overlay[:, :, 1] = 255
            vis = np.where(
                mask[:, :, None] > 127,
                (vis.astype(np.float32) * 0.5 + green_overlay.astype(np.float32) * 0.5).astype(np.uint8),
                vis,
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        cv2.putText(vis, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(vis)
    writer.release()
    print(f"  Saved overlay video: {overlay_path}")

    return overlay_path


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect motion regions in a video and output bboxes on a chosen frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (no camera compensation, temporal decay towards frame 0)
  python motion_bbox_detector.py --video input.mp4

  # Show bboxes on the 10th frame
  python motion_bbox_detector.py --video input.mp4 --display-frame 9

  # Enable camera compensation (for static cameras with shake)
  python motion_bbox_detector.py --video input.mp4 --camera-compensation

  # Detect motion + run SAMURAI segmentation on the full video
  python motion_bbox_detector.py --video input.mp4 --segment

  # Use mean+std thresholding
  python motion_bbox_detector.py --video input.mp4 --threshold-method mean_std

Tuning tips:
  - Wrong region detected?  Try --camera-compensation (or remove it).
    If the camera follows the subject, do NOT use compensation.
  - Too many false detections?  Raise --threshold-percentile (e.g. 90)
  - Missing small objects?  Lower --min-object-area (e.g. 200)
  - Object too spread out?  Lower --temporal-decay (e.g. 0.4)
  - Object position blurred?  Lower --num-frames (e.g. 15)
  - Edge noise?  Raise --border-margin (e.g. 10)
"""
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--num-frames", type=int, default=40,
                        help="Number of frames to load (default: 40)")
    parser.add_argument("--display-frame", type=int, default=0,
                        help="Frame index to draw bboxes on (default: 0)")
    parser.add_argument("--resize", type=int, default=480,
                        help="Resize short edge (default: 480, 0=disable)")
    parser.add_argument("--output-dir", default="motion_output",
                        help="Directory to save results (default: motion_output)")

    # Threshold
    parser.add_argument("--threshold-method", choices=["percentile", "mean_std"],
                        default="percentile", help="Adaptive threshold strategy")
    parser.add_argument("--threshold-percentile", type=float, default=85.0,
                        help="Percentile cutoff (default: 85)")
    parser.add_argument("--threshold-k", type=float, default=2.0,
                        help="Std multiplier for mean_std method (default: 2.0)")

    # Filtering
    parser.add_argument("--min-object-area", type=int, default=400,
                        help="Minimum region area in pixels (default: 400)")
    parser.add_argument("--border-margin", type=int, default=5,
                        help="Reject detections within this margin of frame edge")
    parser.add_argument("--bbox-padding", type=float, default=0.12,
                        help="Pad bboxes by this fraction (default: 0.12)")

    # Morphology
    parser.add_argument("--morph-open-size", type=int, default=5,
                        help="Opening kernel size (default: 5)")
    parser.add_argument("--morph-close-size", type=int, default=21,
                        help="Closing kernel size (default: 21, bridges gaps; 0=disable)")

    # Camera & temporal
    parser.add_argument("--camera-compensation", action="store_true",
                        help="Enable global camera motion subtraction (default: OFF)")
    parser.add_argument("--temporal-decay", type=float, default=0.6,
                        help="Weight decay per frame from display-frame (default: 0.6)")

    # SAMURAI segmentation
    parser.add_argument("--segment", action="store_true",
                        help="After detection, run SAMURAI to segment the strongest "
                             "motion region through the entire video")
    parser.add_argument("--sam-model", choices=["tiny", "small", "base_plus", "large"],
                        default="base_plus", help="SAMURAI model size (default: base_plus)")

    args = parser.parse_args()

    detector = MotionBBoxDetector(
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_k=args.threshold_k,
        min_object_area=args.min_object_area,
        max_area_ratio=0.6,
        morph_open_size=args.morph_open_size,
        morph_close_size=args.morph_close_size,
        camera_compensation=args.camera_compensation,
        temporal_decay=args.temporal_decay,
        border_margin=args.border_margin,
        bbox_padding=args.bbox_padding,
        batch_size=8,
    )

    bboxes, annotated = detector.detect(
        video_path=args.video,
        num_frames=args.num_frames,
        display_frame=args.display_frame,
        resize_short_edge=args.resize if args.resize > 0 else None,
        output_dir=args.output_dir,
    )

    # Run SAMURAI segmentation on the strongest detected region
    if args.segment and bboxes:
        best = bboxes[0]  # sorted by strength, strongest first
        bbox_list = [best.x1, best.y1, best.x2, best.y2]
        resize = args.resize if args.resize > 0 else None

        run_samurai_segmentation(
            video_path=args.video,
            bbox=bbox_list,
            start_frame=args.display_frame,
            model_size=args.sam_model,
            output_dir=args.output_dir,
            resize_short_edge=resize,
        )
    elif args.segment and not bboxes:
        print("\nNo motion regions detected — skipping SAMURAI segmentation.")


if __name__ == "__main__":
    main()
