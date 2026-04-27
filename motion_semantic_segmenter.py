#!/usr/bin/env python3
"""
Motion + Semantic Bounding Box Detector with SAMURAI

Extends motion_bbox_detector.py with a DINOv2-based semantic splitting step.
Close-moving objects (e.g. a horse and its rider, two people walking together)
produce one merged motion blob from optical flow alone. Using DINOv2 dense
patch features, we cluster pixels within each motion blob by *semantic
appearance*, so visually different objects become separate bboxes even when
they move together.

Pipeline:
  A. Motion detection (reused from motion_bbox_detector.MotionBBoxDetector):
     1. RAFT optical flow
     2. Temporal-decay weighted magnitude accumulation (centred on display_frame)
     3. Adaptive threshold + morphological cleanup
     4. Connected components -> per-blob binary masks

  B. DINOv2 semantic splitting (NEW):
     5. Extract dense DINOv2 patch features for the display frame
     6. For each motion blob:
          - Sample pixel coordinates within the blob
          - Look up each pixel's DINOv2 feature (via patch mapping)
          - K-means cluster with k in [1..max_k], select best k via silhouette
          - If silhouette >= threshold, split; else keep the blob as one
     7. Each resulting sub-cluster -> hot-core tightened bounding box

  C. SAMURAI segmentation (one object per bbox):
     8. For each split bbox, run SAMURAI independently with that bbox as
        the initial prompt on display_frame
     9. Combine per-object masks into a single multi-colour overlay video

Why DINOv2?
  DINOv2 produces rich self-supervised features where semantically-distinct
  regions (different object classes, different textures/materials) live in
  different parts of feature space. This works even when colours look similar
  and motion is identical, so it complements optical flow rather than duplicating
  it. No text prompts, no category list, no detection head needed — pure
  unsupervised clustering on a motion-masked region.

Usage:
    python motion_semantic_segmenter.py --video horsejump-high.mp4 --segment
    python motion_semantic_segmenter.py --video horsejump-high.mp4 \
        --num-frames 30 --display-frame 9 --max-k 3 --segment
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from motion_bbox_detector import (  # noqa: E402
    MotionBBox,
    MotionBBoxDetector,
    run_samurai_segmentation,
    SAM2_CONFIGS,
    SAM2_DIR,
)


# ---------------------------------------------------------------------------
#  DINOv2 dense feature extractor
# ---------------------------------------------------------------------------

class DinoFeatureExtractor:
    """
    Thin wrapper around DINOv2 (via torch.hub) that produces a dense
    per-patch feature map for a single frame.

    DINOv2 uses 14x14 pixel patches. For an input resized to (Ht, Wt) where
    Ht % 14 == 0 and Wt % 14 == 0, the output map has shape (Ht/14, Wt/14, D).

    We also provide ``features_at_pixels`` which maps pixel (y, x) coordinates
    in the ORIGINAL frame to their corresponding patch feature vectors.
    """

    # Supported model sizes and their feature dimensions
    _SIZES = {
        "vits14": 384,
        "vitb14": 768,
        "vitl14": 1024,
        "vitg14": 1536,
    }

    def __init__(
        self,
        model_size: str = "vitb14",
        device: Optional[torch.device] = None,
        layers: Optional[List[int]] = None,
    ):
        """
        Args:
            layers: block indices (0-based) to concatenate features from.
                Lower layers capture more texture/appearance; the last layer
                captures high-level semantics. Concatenating several gives
                richer within-object discrimination than a single layer.
                Default: last 4 blocks.
        """
        assert model_size in self._SIZES, f"Unknown DINO size: {model_size}"
        self.model_size = model_size
        self.feat_dim_raw = self._SIZES[model_size]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  Loading DINOv2-{model_size} ...")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{model_size}", verbose=False
        )
        self.model = self.model.eval().to(self.device)

        n_blocks = len(self.model.blocks)
        if layers is None:
            # Mix of mid-depth (texture) and last (semantics)
            layers = [n_blocks // 2, (n_blocks * 3) // 4, n_blocks - 1]
        for li in layers:
            assert 0 <= li < n_blocks, f"layer {li} out of range [0,{n_blocks})"
        self.layers = layers
        self.feat_dim = self.feat_dim_raw * len(layers)

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    @torch.no_grad()
    def extract(
        self, frame_bgr: np.ndarray, max_side: int = 896
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract dense patch features.

        Args:
            frame_bgr: HxWx3 BGR uint8 frame.
            max_side:  resize so the longer side is at most this value before
                       DINO (feature maps are denser if we downscale less;
                       518 is DINOv2's preferred size at inference).

        Returns:
            features: (feat_h, feat_w, D) float32 numpy array on CPU.
            original_shape: (H, W) of the input frame.
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # Pick a target size divisible by 14, respecting max_side
        scale = min(max_side / max(orig_h, orig_w), 1.0)
        tgt_h = int(round(orig_h * scale))
        tgt_w = int(round(orig_w * scale))
        tgt_h = max(14, (tgt_h // 14) * 14)
        tgt_w = max(14, (tgt_w // 14) * 14)

        img = cv2.resize(frame_bgr, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        t = (t - self._mean) / self._std
        t = t.unsqueeze(0)

        # Multi-layer features: each layer is (1, D, H, W)
        layer_feats = self.model.get_intermediate_layers(
            t, n=self.layers, reshape=True, norm=True,
        )
        # Concatenate along channel dim -> (1, D*n_layers, H, W)
        feats = torch.cat(list(layer_feats), dim=1)
        feats = feats[0].permute(1, 2, 0).contiguous().cpu().numpy()  # (H, W, D_total)

        norms = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8
        feats = feats / norms
        return feats.astype(np.float32), (orig_h, orig_w)

    @staticmethod
    def features_at_pixels(
        features: np.ndarray, orig_shape: Tuple[int, int],
        ys: np.ndarray, xs: np.ndarray,
        bilinear: bool = True,
    ) -> np.ndarray:
        """
        Sample the patch feature map at pixel coordinates in the original
        image space. Uses bilinear interpolation by default so features vary
        smoothly across patch boundaries.

        Returns (N, D) L2-normalised feature vectors.
        """
        orig_h, orig_w = orig_shape
        feat_h, feat_w, D = features.shape
        # Map pixel -> continuous patch-grid coords, centred on patch centres
        gy = ys * (feat_h / orig_h) - 0.5
        gx = xs * (feat_w / orig_w) - 0.5

        if not bilinear:
            pi = np.clip(np.round(gy).astype(np.int32), 0, feat_h - 1)
            pj = np.clip(np.round(gx).astype(np.int32), 0, feat_w - 1)
            out = features[pi, pj]
        else:
            y0 = np.clip(np.floor(gy).astype(np.int32), 0, feat_h - 1)
            x0 = np.clip(np.floor(gx).astype(np.int32), 0, feat_w - 1)
            y1 = np.clip(y0 + 1, 0, feat_h - 1)
            x1 = np.clip(x0 + 1, 0, feat_w - 1)
            wy = np.clip(gy - y0, 0.0, 1.0).astype(np.float32)
            wx = np.clip(gx - x0, 0.0, 1.0).astype(np.float32)
            f00 = features[y0, x0]
            f01 = features[y0, x1]
            f10 = features[y1, x0]
            f11 = features[y1, x1]
            wy_ = wy[:, None]
            wx_ = wx[:, None]
            top = f00 * (1 - wx_) + f01 * wx_
            bot = f10 * (1 - wx_) + f11 * wx_
            out = top * (1 - wy_) + bot * wy_

        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return (out / norms).astype(np.float32)


# ---------------------------------------------------------------------------
#  Simple KMeans + silhouette helper (scipy-only; sklearn may be missing)
# ---------------------------------------------------------------------------

def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean silhouette coefficient. O(N^2) in N; keep N <= ~2000.
    Uses Euclidean distance (X is already L2-normalised, so this is
    equivalent up to monotonic transform to cosine distance).
    """
    n = len(X)
    if n < 4:
        return -1.0
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0

    D = cdist(X, X)  # (N, N)

    a = np.zeros(n)
    b = np.full(n, np.inf)
    for c in unique:
        members = labels == c
        n_c = members.sum()
        # a(i) = mean distance to own-cluster points (excluding self)
        if n_c > 1:
            row_sums = D[:, members].sum(axis=1)
            a_c = (row_sums - 0.0) / (n_c - 1)
            a[members] = a_c[members]
        # b(i) from other clusters = min mean distance to any other cluster
        for c2 in unique:
            if c2 == c:
                continue
            others = labels == c2
            if others.sum() == 0:
                continue
            mean_to_other = D[:, others].mean(axis=1)
            b[members] = np.minimum(b[members], mean_to_other[members])

    # Points in singleton clusters get silhouette 0 by convention
    s = np.where(np.maximum(a, b) > 0,
                 (b - a) / np.maximum(a, b),
                 0.0)
    # Ignore singleton clusters in the mean (a stayed 0)
    return float(s.mean())


def _kmeans_best_k(
    X: np.ndarray, max_k: int = 3, min_silhouette: float = 0.15,
    k_penalty: float = 0.03, random_state: int = 42,
) -> Tuple[int, np.ndarray, float]:
    """
    Try k = 2..max_k, pick the k whose silhouette - (k-2)*k_penalty is highest.
    This biases the search toward simpler models unless a higher k improves
    the silhouette by a meaningful margin. If the best adjusted score is below
    ``min_silhouette`` we return k=1 (no split).

    Returns (k, labels, silhouette).
    """
    n = len(X)
    if n < 20:
        return 1, np.zeros(n, dtype=np.int32), -1.0

    # Subsample for silhouette calculation if very large
    if n > 1500:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(n, 1500, replace=False)
        X_s = X[sample_idx]
    else:
        X_s = X

    best_k = 1
    best_adj = -np.inf
    best_raw = -1.0
    best_labels_full = np.zeros(n, dtype=np.int32)

    for k in range(2, max_k + 1):
        if len(X_s) < k * 5:
            break
        try:
            centroids, labels_s = kmeans2(
                X_s, k, minit="++", seed=random_state, iter=30,
            )
        except Exception as e:
            print(f"      k={k}: kmeans2 failed ({e})")
            continue
        if len(np.unique(labels_s)) < k:
            # degenerate — some cluster got no points
            continue
        score = _silhouette(X_s, labels_s)
        adj = score - (k - 2) * k_penalty
        print(f"      k={k}: silhouette={score:+.3f}  adjusted={adj:+.3f}")

        if adj > best_adj:
            best_adj = adj
            best_raw = score
            best_k = k
            # Assign labels to ALL points by nearest centroid
            D = cdist(X, centroids)
            best_labels_full = D.argmin(axis=1).astype(np.int32)

    if best_k == 1 or best_raw < min_silhouette:
        return 1, np.zeros(n, dtype=np.int32), best_raw
    return best_k, best_labels_full, best_raw


# ---------------------------------------------------------------------------
#  SAM2-based object proposer (alternative to DINO clustering)
# ---------------------------------------------------------------------------

class Sam2ObjectProposer:
    """
    Generate per-object masks by prompting SAM2 with a grid of points inside
    each motion blob. SAM2 is purpose-built for object segmentation, so for
    compound scenes like 'horse + rider' it separates them more reliably
    than pure feature clustering.

    Workflow per motion blob:
      1. Sample a regular grid of points inside the blob.
      2. For each point, ask SAM2 for multi-mask candidates.
      3. Keep masks whose pixels lie mostly inside the motion mask.
      4. NMS-deduplicate masks by IoU.

    Compared to SAM2AutomaticMaskGenerator, this version runs only inside
    the motion region (far faster) and post-filters against motion overlap.
    """

    def __init__(
        self,
        sam_model_size: str = "base_plus",
        points_per_side: int = 6,
        pred_iou_thresh: float = 0.70,
        stability_thresh: float = 0.85,
        motion_overlap_thresh: float = 0.60,
        mask_nms_iou: float = 0.60,
        min_mask_area: int = 400,
    ):
        if os.path.isdir(SAM2_DIR):
            sys.path.insert(0, SAM2_DIR)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_thresh = stability_thresh
        self.motion_overlap_thresh = motion_overlap_thresh
        self.mask_nms_iou = mask_nms_iou
        self.min_mask_area = min_mask_area

        config_file, ckpt_name = SAM2_CONFIGS[sam_model_size]
        ckpt_path = os.path.join(SAM2_DIR, "checkpoints", ckpt_name)
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", ckpt_name)

        print(f"  Loading SAM2 image model ({sam_model_size}) ...")
        sam_model = build_sam2(
            config_file=config_file, ckpt_path=ckpt_path, device=self.device,
        )
        self.predictor = SAM2ImagePredictor(sam_model)

    @torch.no_grad()
    def propose(
        self, frame_bgr: np.ndarray, motion_mask: np.ndarray,
        blob_masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Returns a list of per-object binary masks (255=object) across ALL blobs.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []

        for blob_idx, blob in enumerate(blob_masks):
            cand_masks, cand_scores = self._propose_for_blob(blob, motion_mask)
            print(f"     blob {blob_idx}: {len(cand_masks)} SAM2 proposal(s)")
            all_masks.extend(cand_masks)
            all_scores.extend(cand_scores)

        # Final global NMS across all blobs' candidates, ranked by SAM2 score
        return self._nms_masks(all_masks, all_scores, self.mask_nms_iou)

    def _propose_for_blob(
        self, blob_mask: np.ndarray, motion_mask: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float]]:
        ys, xs = np.where(blob_mask > 0)
        if len(ys) < self.min_mask_area:
            return [], []

        # Sample a regular grid of prompt points inside the blob
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1

        pts = []
        step_x = max(1, bw // (self.points_per_side + 1))
        step_y = max(1, bh // (self.points_per_side + 1))
        for i in range(1, self.points_per_side + 1):
            for j in range(1, self.points_per_side + 1):
                px = x1 + j * step_x
                py = y1 + i * step_y
                if 0 <= py < blob_mask.shape[0] and 0 <= px < blob_mask.shape[1]:
                    if blob_mask[py, px] > 0:
                        pts.append((px, py))
        if not pts:
            # Fallback: centroid
            pts = [(int(xs.mean()), int(ys.mean()))]

        # Query SAM2 per point. multimask_output=True returns 3 masks
        # (small/medium/large) plus per-mask IoU scores. For object detection
        # we pick the *best-scoring* mask per point - the "medium-sized"
        # interpretation that usually corresponds to the object under the
        # point, not a part of it (too small) nor a scene-level merger (too
        # large). We also reject any mask whose area exceeds 95% of the
        # motion blob's bounding box (likely the whole compound subject).
        cand_masks: List[np.ndarray] = []
        cand_scores: List[float] = []
        h, w = blob_mask.shape
        blob_bin = blob_mask > 0
        blob_area = int(blob_bin.sum())

        for (px, py) in pts:
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[px, py]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
            masks = np.asarray(masks)
            if masks.ndim == 4:
                masks = masks[:, 0]
            scores = np.asarray(scores).flatten()

            # Pick the best-scoring mask per point that passes sanity checks.
            # SAM2 scores reflect mask quality and don't consistently map to
            # part/object/scene scale, so score-ordered selection is most
            # reliable. Sub-parts (helmet, saddle) that slip through here are
            # removed downstream by the asymmetric bbox-containment NMS,
            # which compares how far the smaller candidate's bbox extends
            # beyond a kept larger candidate's bbox.
            order = np.argsort(-scores)
            for k in order:
                if float(scores[k]) < self.pred_iou_thresh:
                    continue
                m = (masks[k] > 0).astype(np.uint8)
                m_area = int(m.sum())
                if m_area < self.min_mask_area:
                    continue
                if m_area > 0.85 * h * w:
                    continue

                # --- Spatial containment within the motion blob ------------
                # An object mask must:
                #   1. be MOSTLY inside the motion blob (>= self.motion_overlap_thresh)
                #   2. NOT cover 'too much' of the blob (that's the compound)
                #   3. NOT be disproportionately larger than the blob
                m_bin = m > 0
                inter = int(np.logical_and(m_bin, blob_bin).sum())
                frac_in_blob = inter / max(1, m_area)
                frac_of_blob = inter / max(1, blob_area)

                if frac_in_blob < self.motion_overlap_thresh:
                    continue
                if frac_of_blob > 0.9:
                    continue
                if m_area > 2.0 * blob_area:
                    continue

                # Reject thin strips & pole-shaped masks (camera-motion
                # residuals on poles / fences). Real articulated objects
                # are wider than ~30 px in both dims and have an aspect
                # ratio below ~5:1.
                mys, mxs = np.where(m_bin)
                if len(mys) > 0:
                    mw = int(mxs.max() - mxs.min() + 1)
                    mh = int(mys.max() - mys.min() + 1)
                    min_dim = max(30, int(min(h, w) * 0.05))
                    if mw < min_dim or mh < min_dim:
                        continue
                    aspect = max(mw, mh) / max(1, min(mw, mh))
                    if aspect > 5.0:
                        continue
                    # mask "solidity" inside its bbox
                    if m_area / (mw * mh) < 0.25:
                        continue

                cand_masks.append(m.astype(np.uint8) * 255)
                cand_scores.append(float(scores[k]))
                break

        # Per-blob NMS (within-blob duplicates)
        kept_masks = self._nms_masks(cand_masks, cand_scores, self.mask_nms_iou)
        # Recompute scores for kept masks (use mean SAM2 score of originals
        # that overlap - but simpler: sort later globally by area)
        kept_scores = [1.0] * len(kept_masks)
        # Preserve original scores approximately: match each kept to the
        # highest-score original that fully contains it.
        for i, km in enumerate(kept_masks):
            km_bin = km > 0
            km_area = int(km_bin.sum())
            best = 0.0
            for cm, cs in zip(cand_masks, cand_scores):
                if int(np.logical_and(km_bin, cm > 0).sum()) >= 0.95 * km_area:
                    best = max(best, cs)
            kept_scores[i] = best or 1.0
        return kept_masks, kept_scores

    @staticmethod
    def _mask_bbox(m_bin: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(m_bin)
        if len(ys) == 0:
            return 0, 0, 0, 0
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _nms_masks(
        masks: List[np.ndarray], scores: List[float], iou_thresh: float,
        max_ext_px: int = 12,
    ) -> List[np.ndarray]:
        """
        NMS, *largest-mask-first*. A smaller candidate is suppressed if it:
          - has IoU with a kept mask > ``iou_thresh``, or
          - is pixel-contained (>=70%) inside a kept mask with larger area, or
          - is *nested* inside a kept mask's bbox: the smaller candidate's
            bbox extends beyond the kept candidate's bbox by no more than
            ``max_ext_px`` pixels on any side (catches sub-parts like a
            helmet sitting within a rider's bbox, while still preserving
            stacked objects like a rider that clearly extends above the
            horse it stands on).

        Containment checks are ONE-DIRECTIONAL (only the smaller candidate
        can be suppressed), so well-separated objects of similar size both
        survive.
        """
        if not masks:
            return []
        areas = np.asarray([int((m > 0).sum()) for m in masks], dtype=np.int64)
        order = np.argsort(-areas)
        kept: List[np.ndarray] = []
        kept_bin: List[np.ndarray] = []
        kept_area: List[int] = []
        kept_bbox: List[Tuple[int, int, int, int]] = []
        for idx in order:
            m_bin = masks[idx] > 0
            m_area = int(m_bin.sum())
            if m_area == 0:
                continue
            bx1, by1, bx2, by2 = Sam2ObjectProposer._mask_bbox(m_bin)
            dup = False
            for k_bin, k_area, k_bbox in zip(kept_bin, kept_area, kept_bbox):
                inter = int(np.logical_and(m_bin, k_bin).sum())
                union = int(np.logical_or(m_bin, k_bin).sum())
                iou = inter / union if union > 0 else 0.0
                if iou > iou_thresh:
                    dup = True
                    break
                if m_area < k_area and inter / max(1, m_area) > 0.7:
                    dup = True
                    break
                # Bbox-extension containment: how far does the smaller
                # candidate's bbox stick out beyond the kept candidate's
                # bbox on each side?  If all 4 sticks-out are small (a few
                # pixels), the candidate is effectively nested inside the
                # kept mask and is a sub-part.
                kx1, ky1, kx2, ky2 = k_bbox
                if m_area < k_area:
                    ext_left   = max(0, kx1 - bx1)
                    ext_top    = max(0, ky1 - by1)
                    ext_right  = max(0, bx2 - kx2)
                    ext_bottom = max(0, by2 - ky2)
                    if max(ext_left, ext_top, ext_right, ext_bottom) <= max_ext_px:
                        dup = True
                        break
            if not dup:
                kept.append(masks[idx])
                kept_bin.append(m_bin)
                kept_area.append(m_area)
                kept_bbox.append((bx1, by1, bx2, by2))
        return kept


# ---------------------------------------------------------------------------
#  Semantic motion detector
# ---------------------------------------------------------------------------

class SemanticMotionDetector:
    """
    Runs motion detection, then splits each motion blob into semantically
    coherent sub-regions using DINOv2 features.

    Wraps a MotionBBoxDetector for the flow/accumulation/mask stages and
    adds the DINO clustering stage.
    """

    def __init__(
        self,
        motion_detector: MotionBBoxDetector,
        split_method: str = "sam2",
        dino_model_size: str = "vitb14",
        max_k: int = 3,
        min_silhouette: float = 0.15,
        k_penalty: float = 0.03,
        spatial_weight: float = 0.25,
        cluster_min_area: int = 300,
        cluster_morph_size: int = 7,
        dino_max_side: int = 896,
        sam_model_size: str = "base_plus",
        sam_points_per_side: int = 6,
        sam_motion_overlap: float = 0.6,
    ):
        """
        Args:
            split_method: "sam2" (default) or "dino".
                sam2:  prompt SAM2 with a grid of points inside each motion
                       blob and keep proposals well-aligned with the motion
                       mask. More reliable for adjacent objects (e.g.
                       horse+rider) because SAM2 was trained to separate
                       distinct objects.
                dino:  cluster blob pixels by DINOv2 feature similarity
                       (+ optional spatial coordinates). Pure unsupervised
                       clustering, no SAM needed for the detection stage.
            spatial_weight: DINO only. How strongly to regularise clustering
                toward spatial coherence. 0 = pure appearance.
            k_penalty: DINO only. Penalty added per extra cluster to favour
                simpler models.
            dino_max_side: DINO only. Max side fed to the DINOv2 backbone.
        """
        assert split_method in {"sam2", "dino"}
        self.md = motion_detector
        self.split_method = split_method
        self.max_k = max_k
        self.min_silhouette = min_silhouette
        self.k_penalty = k_penalty
        self.spatial_weight = spatial_weight
        self.cluster_min_area = cluster_min_area
        self.cluster_morph_size = cluster_morph_size
        self.dino_max_side = dino_max_side

        self.dino: Optional[DinoFeatureExtractor] = None
        self.sam: Optional[Sam2ObjectProposer] = None
        if split_method == "dino":
            self.dino = DinoFeatureExtractor(model_size=dino_model_size)
        else:
            self.sam = Sam2ObjectProposer(
                sam_model_size=sam_model_size,
                points_per_side=sam_points_per_side,
                motion_overlap_thresh=sam_motion_overlap,
                min_mask_area=cluster_min_area,
            )

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        video_path: str,
        num_frames: int = 40,
        display_frame: int = 0,
        resize_short_edge: Optional[int] = 480,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[MotionBBox], np.ndarray, np.ndarray]:
        """
        Returns:
            bboxes:       list of semantically-split MotionBBox
            annotated:    display frame with bboxes + heatmap overlay
            display_frame_bgr: the raw display frame (no annotations)
        """
        t0 = time.time()
        md = self.md

        # --- Stage A: frames + flow + heatmap ------------------------------
        print(f"[A1] Loading up to {num_frames} frames from {video_path} ...")
        frames = md._load_frames(video_path, num_frames, resize_short_edge)
        if len(frames) < 3:
            print("Need at least 3 frames. Aborting.")
            return [], frames[0] if frames else np.zeros((1, 1, 3), np.uint8), frames[0]

        h, w = frames[0].shape[:2]
        print(f"     {len(frames)} frames at {w}x{h}")

        print(f"[A2] Computing RAFT optical flow ({len(frames) - 1} pairs) ...")
        tf = time.time()
        flows = md._compute_flows(frames)
        print(f"     Done in {time.time() - tf:.1f}s")

        magnitudes = []
        for flow in flows:
            if md.camera_compensation:
                flow = md._compensate_camera(flow)
            magnitudes.append(md._compute_magnitude(flow))

        anchor = min(display_frame, len(magnitudes) - 1)
        print(f"[A3] Accumulating with temporal decay={md.temporal_decay} "
              f"centred on frame {anchor} ...")
        heatmap = md._accumulate_magnitudes(magnitudes, anchor)

        print(f"[A4] Adaptive threshold ({md.threshold_method}) + morphology ...")
        mask = md._adaptive_threshold(heatmap)
        mask = md._refine_mask(mask)

        # RAFT pads to multiples of 8, so mask/heatmap may be a few px smaller
        # than the original frames. Resize both to match display resolution
        # so all downstream coords are consistent.
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        n_active = int(np.count_nonzero(mask))
        print(f"     Active pixels: {n_active} ({n_active / (h * w):.1%} of frame)")

        # --- Stage B: per-blob semantic splitting --------------------------
        display = frames[min(display_frame, len(frames) - 1)].copy()

        # Collect motion blobs
        print(f"[B1] Finding motion blobs ...")
        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        blob_masks_raw: List[np.ndarray] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < md.min_object_area:
                continue
            if area > h * w * md.max_area_ratio:
                continue
            blob_masks_raw.append((labels_map == i).astype(np.uint8) * 255)
        print(f"     {len(blob_masks_raw)} motion blob(s)")

        final_masks: List[np.ndarray] = []
        if self.split_method == "dino":
            print(f"[B2] DINOv2-{self.dino.model_size} semantic splitting ...")
            tfeat = time.time()
            features, orig_shape = self.dino.extract(display, max_side=self.dino_max_side)
            print(f"     DINO feature map {features.shape} in "
                  f"{time.time() - tfeat:.1f}s")
            for i, blob in enumerate(blob_masks_raw):
                print(f"  Blob {i}: area={int((blob > 0).sum())}px")
                sub_masks = self._split_blob_semantically(blob, features, orig_shape)
                print(f"     -> {len(sub_masks)} sub-region(s)")
                final_masks.extend(sub_masks)
        else:
            print(f"[B2] SAM2 object-proposal splitting ...")
            tfeat = time.time()
            proposals = self.sam.propose(display, mask, blob_masks_raw)
            print(f"     SAM2 proposed {len(proposals)} object(s) in "
                  f"{time.time() - tfeat:.1f}s")
            if proposals:
                final_masks = proposals
            else:
                # Fallback: keep raw blobs
                print(f"     (no proposals survived filtering — using raw blobs)")
                final_masks = blob_masks_raw

        # --- Stage B3: bbox extraction from each final mask ----------------
        bboxes = self._masks_to_bboxes(final_masks, heatmap, h, w)
        bboxes = md._nms(bboxes, iou_threshold=0.3)
        print(f"[B3] Final object count after semantic split + NMS: {len(bboxes)}")

        # --- Visualise ------------------------------------------------------
        annotated = md._visualize(display, bboxes, heatmap)

        # Save outputs
        if output_dir:
            self._save_outputs(output_dir, annotated, mask, heatmap,
                               bboxes, display_frame, final_masks, display)

        for i, bb in enumerate(bboxes):
            print(f"  Obj {i}: bbox=[{bb.x1}, {bb.y1}, {bb.x2}, {bb.y2}]  "
                  f"size={bb.width}x{bb.height}  strength={bb.motion_strength:.2f}")
        print(f"[done] total time: {time.time() - t0:.1f}s")

        return bboxes, annotated, display

    # ------------------------------------------------------------------ #
    #  Semantic blob splitting                                            #
    # ------------------------------------------------------------------ #

    def _build_feature_vectors(
        self, ys: np.ndarray, xs: np.ndarray,
        features: np.ndarray, orig_shape: Tuple[int, int],
        blob_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Concatenate L2-normalised DINO features with spatial coords (scaled
        by ``spatial_weight``). Spatial coords are normalised by the blob's
        own bounding box so weight is interpretable independent of blob size.
        """
        feats = DinoFeatureExtractor.features_at_pixels(
            features, orig_shape, ys, xs,
        )
        if self.spatial_weight <= 0:
            return feats
        bx1, by1, bx2, by2 = blob_bbox
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        sx = ((xs - bx1) / bw).astype(np.float32)
        sy = ((ys - by1) / bh).astype(np.float32)
        coord = np.stack([sx, sy], axis=1) * self.spatial_weight
        return np.concatenate([feats, coord.astype(np.float32)], axis=1)

    def _split_blob_semantically(
        self, blob_mask: np.ndarray, features: np.ndarray,
        orig_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """
        Split a single motion blob using DINOv2 feature clustering
        with optional spatial regularisation.
        """
        ys, xs = np.where(blob_mask > 0)
        n_pix = len(ys)
        if n_pix < self.cluster_min_area:
            return [blob_mask]

        blob_bbox = (int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1)

        # Subsample up to 2000 pixels for clustering
        if n_pix > 2000:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_pix, 2000, replace=False)
            ys_s = ys[sample_idx]
            xs_s = xs[sample_idx]
        else:
            ys_s = ys
            xs_s = xs

        feats_s = self._build_feature_vectors(ys_s, xs_s, features, orig_shape, blob_bbox)

        k, labels_s, score = _kmeans_best_k(
            feats_s, max_k=self.max_k,
            min_silhouette=self.min_silhouette,
            k_penalty=self.k_penalty,
        )

        if k == 1:
            return [blob_mask]

        # Propagate cluster labels to ALL blob pixels by nearest centroid
        # (centroids use the SAME feature format: dino + spatial)
        centroids = np.stack([feats_s[labels_s == c].mean(axis=0) for c in range(k)])

        feats_all = self._build_feature_vectors(ys, xs, features, orig_shape, blob_bbox)
        D_all = cdist(feats_all, centroids)
        labels_all = D_all.argmin(axis=1)

        # Build per-cluster masks, clean them up, discard tiny ones
        out: List[np.ndarray] = []
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cluster_morph_size, self.cluster_morph_size),
        )
        for c in range(k):
            m = np.zeros_like(blob_mask)
            pix = labels_all == c
            if pix.sum() < self.cluster_min_area:
                continue
            m[ys[pix], xs[pix]] = 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

            # A single cluster may fragment into several connected components;
            # keep the largest one per cluster to avoid speckle.
            n_sub, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_sub <= 1:
                continue
            largest = 1 + int(np.argmax(sub_stats[1:, cv2.CC_STAT_AREA]))
            largest_area = int(sub_stats[largest, cv2.CC_STAT_AREA])
            if largest_area < self.cluster_min_area:
                continue
            m_clean = (sub_labels == largest).astype(np.uint8) * 255
            out.append(m_clean)

        if len(out) < 2:
            # Splitting didn't survive cleanup — keep the original blob
            return [blob_mask]
        return out

    # ------------------------------------------------------------------ #
    #  Mask -> bbox (hot-core tightening, same as parent class)           #
    # ------------------------------------------------------------------ #

    def _masks_to_bboxes(
        self, masks: List[np.ndarray], heatmap: np.ndarray, h: int, w: int,
    ) -> List[MotionBBox]:
        md = self.md
        out: List[MotionBBox] = []
        for m in masks:
            ys, xs = np.where(m > 0)
            if len(ys) < md.min_object_area:
                continue

            area = int(len(ys))
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            if bw < max(20, int(min(h, w) * 0.04)) or bh < max(20, int(min(h, w) * 0.04)):
                continue

            # Hot-core tightening
            component_mask = m > 0
            comp_mags = heatmap[component_mask]
            local_thresh = float(np.percentile(comp_mags, 60))
            hot_mask = component_mask & (heatmap >= local_thresh)
            hys, hxs = np.where(hot_mask)
            if len(hys) > 0:
                x1, y1 = int(hxs.min()), int(hys.min())
                x2, y2 = int(hxs.max()) + 1, int(hys.max()) + 1
            else:
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1

            # Pad
            bw_ = x2 - x1
            bh_ = y2 - y1
            px = int(bw_ * md.bbox_padding)
            py = int(bh_ * md.bbox_padding)
            x1 = max(0, x1 - px)
            y1 = max(0, y1 - py)
            x2 = min(w, x2 + px)
            y2 = min(h, y2 + py)

            strength = float(heatmap[component_mask].mean())
            out.append(MotionBBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                area=area, motion_strength=strength, frame_presence=strength,
            ))

        out.sort(key=lambda b: b.motion_strength, reverse=True)
        return out

    # ------------------------------------------------------------------ #
    #  Output saving                                                      #
    # ------------------------------------------------------------------ #

    def _save_outputs(
        self, output_dir: str, annotated: np.ndarray, mask: np.ndarray,
        heatmap: np.ndarray, bboxes: List[MotionBBox], display_frame: int,
        cluster_masks: List[np.ndarray], display_bgr: np.ndarray,
    ):
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir, "motion_bboxes.png"), annotated)
        cv2.imwrite(os.path.join(output_dir, "motion_mask.png"), mask)

        heatmap_vis = cv2.normalize(heatmap, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, "motion_heatmap.png"), heatmap_vis)

        # Colour-coded cluster mask (one colour per semantic sub-region)
        if cluster_masks:
            disp_h, disp_w = display_bgr.shape[:2]
            cluster_vis = display_bgr.copy()
            colours = _colour_palette(len(cluster_masks))
            overlay = np.zeros_like(display_bgr)
            for m, c in zip(cluster_masks, colours):
                if m.shape[:2] != (disp_h, disp_w):
                    m = cv2.resize(m, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                overlay[m > 0] = c
            cluster_vis = cv2.addWeighted(cluster_vis, 0.55, overlay, 0.45, 0)
            cv2.imwrite(os.path.join(output_dir, "semantic_clusters.png"), cluster_vis)

        with open(os.path.join(output_dir, "bboxes.txt"), "w") as f:
            f.write(f"# display_frame={display_frame}\n")
            f.write(f"# format: obj_id x1 y1 x2 y2 area strength\n")
            for i, bb in enumerate(bboxes):
                f.write(f"{i} {bb.x1} {bb.y1} {bb.x2} {bb.y2} "
                        f"{bb.area} {bb.motion_strength:.4f}\n")

        print(f"\n  Saved detection outputs to {output_dir}/")


# ---------------------------------------------------------------------------
#  Multi-object SAMURAI segmentation
# ---------------------------------------------------------------------------

def _colour_palette(n: int) -> List[Tuple[int, int, int]]:
    """Distinct BGR colours for up to n objects."""
    base = [
        (0, 255, 0),     # green
        (255, 100, 0),   # orange-ish
        (0, 100, 255),   # red
        (255, 0, 255),   # magenta
        (0, 255, 255),   # yellow
        (255, 255, 0),   # cyan
        (128, 255, 0),   # lime
        (255, 128, 0),   # blue-orange
    ]
    return [base[i % len(base)] for i in range(n)]


def run_samurai_multi(
    video_path: str,
    bboxes: List[MotionBBox],
    start_frame: int,
    output_dir: str,
    model_size: str = "base_plus",
    resize_short_edge: Optional[int] = 480,
) -> str:
    """
    Run SAMURAI once per bbox and combine per-object masks into a single
    colour-coded overlay video.

    Each object's masks go to ``<output_dir>/obj_<i>/masks/``; the combined
    overlay is written to ``<output_dir>/segmentation_multi.mp4``.
    """
    if len(bboxes) == 0:
        print("No bboxes to segment.")
        return ""

    per_obj_mask_dirs: List[str] = []
    for i, bb in enumerate(bboxes):
        print(f"\n--- SAMURAI object {i + 1}/{len(bboxes)}: "
              f"bbox=[{bb.x1}, {bb.y1}, {bb.x2}, {bb.y2}] ---")
        obj_dir = os.path.join(output_dir, f"obj_{i}")
        run_samurai_segmentation(
            video_path=video_path,
            bbox=bb.to_list(),
            start_frame=start_frame,
            model_size=model_size,
            output_dir=obj_dir,
            resize_short_edge=resize_short_edge,
        )
        per_obj_mask_dirs.append(os.path.join(obj_dir, "masks"))

    # --- Combine overlays ---------------------------------------------------
    print("\nBuilding combined overlay video ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Load all video frames, matching the resize applied to SAMURAI masks
    frames: List[np.ndarray] = []
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
    colours = _colour_palette(len(bboxes))

    out_path = os.path.join(output_dir, "segmentation_multi.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    from tqdm import tqdm
    for fi in tqdm(range(n_frames), desc="  compositing"):
        vis = frames[fi].copy()
        for obj_idx, mask_dir in enumerate(per_obj_mask_dirs):
            m_path = os.path.join(mask_dir, f"{fi:05d}.png")
            if not os.path.isfile(m_path):
                continue
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is None or m.max() == 0:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

            colour = colours[obj_idx]
            coloured = np.zeros_like(vis)
            coloured[:, :] = colour
            sel = m > 127
            vis[sel] = (vis[sel].astype(np.float32) * 0.45 +
                        coloured[sel].astype(np.float32) * 0.55).astype(np.uint8)

            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, colour, 2)

        cv2.putText(vis, f"Frame {fi}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(vis)
    writer.release()
    print(f"Saved multi-object overlay: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect motion regions, semantically split close-moving objects "
            "using DINOv2 features, and optionally run SAMURAI on each."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect only (no SAMURAI): outputs motion_bboxes.png + semantic_clusters.png
  python motion_semantic_segmenter.py --video horsejump-high.mp4

  # Full pipeline: one SAMURAI mask track per split bbox
  python motion_semantic_segmenter.py --video horsejump-high.mp4 --segment

  # Allow up to 4 clusters per blob; be a bit stricter about splitting
  python motion_semantic_segmenter.py --video horsejump-high.mp4 --segment \\
      --max-k 4 --min-silhouette 0.18
""",
    )

    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-dir", default="motion_semantic_output",
                        help="Directory for outputs (default: motion_semantic_output)")

    # Motion detection
    parser.add_argument("--num-frames", type=int, default=40,
                        help="How many frames to use for motion analysis")
    parser.add_argument("--display-frame", type=int, default=0,
                        help="Frame index for bbox output (default: 0)")
    parser.add_argument("--resize", type=int, default=480,
                        help="Resize short edge (default 480; 0 = keep original)")
    parser.add_argument("--camera-compensation", action="store_true",
                        help="Enable RANSAC global-motion subtraction")

    parser.add_argument("--threshold-method", choices=["percentile", "mean_std"],
                        default="percentile")
    parser.add_argument("--threshold-percentile", type=float, default=85.0)
    parser.add_argument("--threshold-k", type=float, default=2.0)
    parser.add_argument("--threshold-floor", type=float, default=0.5,
                        help="Minimum absolute flow magnitude (px) treated "
                             "as motion. Lower for tiny/fast objects whose "
                             "accumulated weighted magnitude falls below 0.5.")
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--max-area-ratio", type=float, default=0.6)
    parser.add_argument("--max-axis-ratio", type=float, default=0.95)
    parser.add_argument("--morph-open", type=int, default=5)
    parser.add_argument("--morph-close", type=int, default=21)
    parser.add_argument("--temporal-decay", type=float, default=0.6)
    parser.add_argument("--bbox-padding", type=float, default=0.12)

    # Semantic split
    parser.add_argument("--split-method", default="sam2", choices=["sam2", "dino"],
                        help="Method for separating close-moving objects. "
                             "'sam2' (default) uses SAM2 point-prompting to "
                             "find distinct objects inside the motion mask "
                             "and is more reliable for adjacent subjects "
                             "(horse+rider). 'dino' clusters pixels by "
                             "DINOv2 feature similarity.")
    parser.add_argument("--dino-model", default="vitb14",
                        choices=["vits14", "vitb14", "vitl14"],
                        help="DINOv2 backbone size (default vitb14)")
    parser.add_argument("--sam-points-per-side", type=int, default=10,
                        help="Grid density for SAM2 prompt points inside "
                             "each motion blob (default 10 -> up to 100 points)")
    parser.add_argument("--sam-motion-overlap", type=float, default=0.7,
                        help="Minimum fraction of a SAM2 mask's pixels that "
                             "must lie inside the motion blob (default 0.7)")
    parser.add_argument("--max-k", type=int, default=3,
                        help="Max number of semantic clusters per motion blob")
    parser.add_argument("--min-silhouette", type=float, default=0.15,
                        help="Minimum silhouette score required to split a blob")
    parser.add_argument("--k-penalty", type=float, default=0.03,
                        help="Per-extra-cluster silhouette penalty; "
                             "favours simpler models (default 0.03)")
    parser.add_argument("--spatial-weight", type=float, default=0.25,
                        help="Weight on (x,y) features during clustering; "
                             "0 = DINO only, higher = more spatially coherent clusters")
    parser.add_argument("--dino-max-side", type=int, default=896,
                        help="Max side (px) fed to DINOv2 (default 896)")

    # SAMURAI
    parser.add_argument("--segment", action="store_true",
                        help="Run SAMURAI on each split bbox")
    parser.add_argument("--sam-model", default="base_plus",
                        choices=["tiny", "small", "base_plus", "large"])

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    resize = args.resize if args.resize > 0 else None

    detector = MotionBBoxDetector(
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_k=args.threshold_k,
        min_object_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        max_axis_ratio=args.max_axis_ratio,
        morph_open_size=args.morph_open,
        morph_close_size=args.morph_close,
        camera_compensation=args.camera_compensation,
        temporal_decay=args.temporal_decay,
        bbox_padding=args.bbox_padding,
        threshold_floor=args.threshold_floor,
    )

    semantic = SemanticMotionDetector(
        motion_detector=detector,
        split_method=args.split_method,
        dino_model_size=args.dino_model,
        max_k=args.max_k,
        min_silhouette=args.min_silhouette,
        k_penalty=args.k_penalty,
        spatial_weight=args.spatial_weight,
        dino_max_side=args.dino_max_side,
        sam_model_size=args.sam_model,
        sam_points_per_side=args.sam_points_per_side,
        sam_motion_overlap=args.sam_motion_overlap,
    )

    bboxes, annotated, _display = semantic.detect(
        video_path=args.video,
        num_frames=args.num_frames,
        display_frame=args.display_frame,
        resize_short_edge=resize,
        output_dir=args.output_dir,
    )

    if args.segment and len(bboxes) > 0:
        run_samurai_multi(
            video_path=args.video,
            bboxes=bboxes,
            start_frame=args.display_frame,
            output_dir=args.output_dir,
            model_size=args.sam_model,
            resize_short_edge=resize,
        )


if __name__ == "__main__":
    main()
