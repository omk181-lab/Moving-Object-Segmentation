# Motion-Aware Video Segmentation Pipeline — Full Implementation Blueprint

## Overview

Build a Python pipeline that automatically detects moving objects in a video using optical flow, then tracks and segments them with pixel-level masks using SAMURAI (motion-aware SAM2). No manual prompts needed — the system discovers what's moving and tracks it end-to-end.

**Target hardware:** NVIDIA A6000 (48GB VRAM)
**Target performance:** ~15-16 seconds for 200 frames, 3-4 objects
**Input:** A video file (MP4) or a folder of JPEG frames
**Output:** Per-frame segmentation masks for each detected moving object, plus visualization overlays

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    PASS 1                            │
│              Optical Flow (RAFT)                     │
│                                                      │
│  Video frames ──► RAFT-Small (batch) ──► Flow maps  │
│  (200 frames)     GPU batched              (199 maps)│
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    PASS 2                            │
│           Motion Analysis (CPU)                      │
│                                                      │
│  Flow maps ──► Magnitude threshold                   │
│            ──► Morphological cleanup                 │
│            ──► Connected components                  │
│            ──► Temporal consistency filter            │
│            ──► Top-K object bboxes on first          │
│               motion frame                           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    PASS 3                            │
│         SAMURAI Tracking + Segmentation              │
│                                                      │
│  Video frames + bboxes                               │
│     ──► SAM2 image encoder (ONCE per frame)          │
│     ──► SAM2 mask decoder (per object, lightweight)  │
│     ──► Kalman filter motion prediction (per object) │
│     ──► Motion-aware memory selection                │
│     ──► Output: per-object binary masks              │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
motion_seg_pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── config.yaml                    # All configurable parameters
│
├── pipeline.py                    # Main entry point - runs all 3 passes
│
├── pass1_optical_flow/
│   ├── __init__.py
│   ├── flow_estimator.py          # RAFT-Small wrapper with batch inference
│   └── flow_utils.py              # Flow visualization, I/O helpers
│
├── pass2_motion_analysis/
│   ├── __init__.py
│   ├── motion_detector.py         # Flow → moving object bboxes
│   └── bbox_utils.py              # NMS, merging, filtering utilities
│
├── pass3_tracking/
│   ├── __init__.py
│   ├── samurai_tracker.py         # Modified SAMURAI with shared encoder
│   └── kalman_filter.py           # Kalman filter for motion prediction
│
├── utils/
│   ├── __init__.py
│   ├── video_io.py                # Video/frame loading and saving
│   ├── visualization.py           # Mask overlay rendering
│   └── timer.py                   # Performance profiling
│
├── outputs/                       # Results go here
│   ├── masks/                     # Per-object binary masks
│   ├── visualizations/            # Overlay videos/frames
│   └── metadata.json              # Detected objects, bboxes, timing
│
└── third_party/
    ├── RAFT/                      # Clone from https://github.com/princeton-vl/RAFT
    └── samurai/                   # Clone from https://github.com/yangchris11/samurai
```

---

## Detailed Implementation — config.yaml

```yaml
# config.yaml — all tunable parameters in one place

video:
  input_path: "input_video.mp4"      # MP4 file or directory of JPEGs
  max_frames: null                    # null = process all frames
  resize_short_edge: 480             # Resize for speed; null = original resolution

pass1_optical_flow:
  model: "raft-small"                # Options: raft-small, raft-things
  weights: "things"                  # Pretrained weights
  batch_size: 8                      # Number of frame pairs per GPU batch
  iters: 12                          # RAFT iteration count (12 = good speed/quality)
  mixed_precision: true              # FP16 for speed

pass2_motion_analysis:
  flow_magnitude_threshold: 2.0      # Pixels with flow magnitude > this are "moving"
  min_object_area: 500               # Minimum pixel area to count as an object
  max_objects: 4                     # Maximum number of objects to track
  morphology_kernel_size: 7          # Erosion/dilation kernel for cleanup
  temporal_window: 5                 # Number of consecutive frames an object must appear
  nms_iou_threshold: 0.3            # NMS threshold for merging overlapping detections
  detection_frame_range: [0, 30]     # Analyze first N frames to find objects
  camera_motion_compensation: true   # Subtract dominant motion (for moving cameras)

pass3_tracking:
  sam2_model_size: "base_plus"       # Options: tiny, small, base_plus, large
  sam2_checkpoint: null              # Auto-downloads if null
  max_objects: 4                     # Must match pass2
  # Kalman filter parameters
  kalman_process_noise: 1.0
  kalman_measurement_noise: 10.0
  # SAMURAI memory selection
  memory_bank_size: 6               # Max frames in memory bank
  motion_score_weight: 0.4
  affinity_score_weight: 0.4
  occurrence_score_weight: 0.2
  memory_quality_threshold: 0.6     # Below this, frame is excluded from memory

output:
  save_masks: true                   # Save per-object binary masks
  save_visualization: true           # Save overlay video
  save_flow: false                   # Save optical flow maps (debug)
  output_dir: "outputs/"
  visualization_alpha: 0.4          # Mask overlay transparency
```

---

## Detailed Implementation — Pass 1: Optical Flow

### File: pass1_optical_flow/flow_estimator.py

```
Class: RAFTFlowEstimator

Constructor:
    - Load RAFT-Small model with pretrained weights
    - Move to GPU, set eval mode
    - Enable torch.cuda.amp for mixed precision

Method: compute_all_flows(frames: List[np.ndarray]) -> List[np.ndarray]
    """
    Compute optical flow for all consecutive frame pairs.
    
    Input: List of N frames as numpy arrays (H, W, 3) in uint8 BGR
    Output: List of N-1 flow maps as numpy arrays (H, W, 2) — dx, dy per pixel
    
    Implementation:
    1. Preprocess frames:
       - Convert BGR to RGB
       - Resize to match config resize_short_edge (maintain aspect ratio)
       - Pad to dimensions divisible by 8 (RAFT requirement)
       - Normalize to [0, 1] float32
       - Stack into tensor (N, 3, H, W)
    
    2. Create frame pairs:
       - pairs = [(frame[0], frame[1]), (frame[1], frame[2]), ..., (frame[N-2], frame[N-1])]
    
    3. Batch inference:
       - Process pairs in batches of config.batch_size
       - For each batch:
           with torch.no_grad(), torch.cuda.amp.autocast():
               _, flow = raft_model(img1_batch, img2_batch, iters=config.iters, test_mode=True)
       - flow shape: (batch, 2, H, W) — 2 channels are (dx, dy)
    
    4. Post-process:
       - Remove padding
       - Resize flow back to original resolution (scale flow values proportionally)
       - Convert to numpy float32
       - Return list of flow maps
    """

Method: compute_flow_magnitude(flow: np.ndarray) -> np.ndarray
    """
    Input: flow map (H, W, 2)
    Output: magnitude map (H, W) — sqrt(dx^2 + dy^2)
    """
```

### RAFT Setup Notes:
```
# Clone RAFT repository
git clone https://github.com/princeton-vl/RAFT.git third_party/RAFT

# Download pretrained weights
# raft-small.pth from: https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT

# The RAFT-Small model has ~1M parameters and runs ~80 FPS on A100
# On A6000, expect ~60 FPS for single pairs, ~45 FPS amortized with batching overhead
```

---

## Detailed Implementation — Pass 2: Motion Analysis

### File: pass2_motion_analysis/motion_detector.py

```
Class: MotionDetector

Method: detect_moving_objects(flows: List[np.ndarray], frames: List[np.ndarray]) -> List[Dict]
    """
    Analyze optical flow to find consistently moving objects and return bounding boxes.
    
    Input:
        flows: List of N-1 flow maps (H, W, 2)
        frames: List of N frames (H, W, 3) — only needed for visualization
    
    Output:
        List of dicts, one per detected object:
        [
            {
                "object_id": 0,
                "initial_bbox": [x1, y1, x2, y2],  # SAM2 format
                "first_frame_idx": 3,                # Frame where object first appears moving
                "confidence": 0.87,
                "avg_motion_magnitude": 12.5
            },
            ...
        ]
        Maximum 4 objects (config.max_objects).
    
    Implementation steps:
    
    STEP 1: Camera motion compensation (if enabled)
        - For each flow map, estimate dominant motion using RANSAC on sparse flow vectors
        - Fit an affine transform (or homography) to the flow field
        - Subtract the estimated camera motion from the flow
        - This leaves only independent object motion
        
        Pseudocode:
            For each flow map:
                # Sample sparse grid of flow vectors
                grid_points = uniform_grid(H, W, step=20)
                src_points = grid_points
                dst_points = grid_points + flow[grid_points]
                
                # RANSAC affine estimation
                affine_matrix, inliers = cv2.estimateAffine2D(
                    src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                
                # Compute camera-induced flow at every pixel
                camera_flow = apply_affine_to_grid(affine_matrix, H, W)
                
                # Subtract to get object-only motion
                object_flow = flow - camera_flow
    
    STEP 2: Flow magnitude thresholding
        - For each (camera-compensated) flow map:
            magnitude = sqrt(dx^2 + dy^2)
            motion_mask = (magnitude > config.flow_magnitude_threshold).astype(uint8)
    
    STEP 3: Morphological cleanup
        - For each motion_mask:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morphology_kernel_size, config.morphology_kernel_size))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    STEP 4: Connected component analysis
        - For each cleaned motion_mask:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask)
            
            # Filter by minimum area
            valid_components = []
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= config.min_object_area:
                    bbox = [stats[i, cv2.CC_STAT_LEFT],
                            stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]]
                    valid_components.append({
                        "bbox": bbox,
                        "area": area,
                        "centroid": centroids[i],
                        "frame_idx": frame_idx
                    })
    
    STEP 5: Temporal consistency filtering
        - Track detected components across config.detection_frame_range
        - Group detections across frames by IoU overlap (> 0.3)
        - Only keep objects that appear in at least config.temporal_window consecutive frames
        - This eliminates transient noise, flickering detections, etc.
        
        Pseudocode:
            object_tracks = []  # List of lists of detections across frames
            
            For each frame's detections:
                For each detection:
                    matched = False
                    For each existing track:
                        if IoU(detection.bbox, track[-1].bbox) > config.nms_iou_threshold:
                            track.append(detection)
                            matched = True
                            break
                    if not matched:
                        object_tracks.append([detection])
            
            # Filter tracks by minimum temporal length
            stable_tracks = [t for t in object_tracks if len(t) >= config.temporal_window]
    
    STEP 6: Select top-K objects
        - Sort stable tracks by average motion magnitude (highest motion first)
        - Take top config.max_objects tracks
        - For each selected track, use the bbox from the first frame of that track
        - Apply NMS across selected bboxes to remove duplicates
        
    STEP 7: Format output
        - For each selected object, create the output dict with:
            - initial_bbox in [x1, y1, x2, y2] format (SAM2's expected format)
            - first_frame_idx: the frame index where this object starts moving
            - confidence: average motion magnitude normalized
    """
```

---

## Detailed Implementation — Pass 3: SAMURAI Tracking

### File: pass3_tracking/samurai_tracker.py

This is the most critical file. It modifies SAMURAI to support multi-object tracking with a SHARED image encoder.

```
Class: SAMURAIMultiObjectTracker

Constructor:
    """
    1. Load SAM2 model with config.sam2_model_size checkpoint
       - Use sam2.build_sam.build_sam2_video_predictor()
       - The video predictor already supports shared encoder + multi-object
    
    2. Initialize Kalman filters (one per object)
       - State: [x_center, y_center, width, height, vx, vy, vw, vh]
       - Each filter tracks position + velocity of one object's bbox
    
    3. Initialize memory banks (one per object)
       - Each object gets its own memory bank of size config.memory_bank_size
       - Stores: frame features, mask features, quality scores
    """

Method: track_objects(
    frames: List[np.ndarray],
    initial_detections: List[Dict]  # Output from Pass 2
) -> Dict[int, List[np.ndarray]]
    """
    Track all detected objects through the entire video.
    
    Input:
        frames: List of N frames (H, W, 3) as numpy uint8 BGR
        initial_detections: List of dicts from Pass 2, each with:
            - object_id: int
            - initial_bbox: [x1, y1, x2, y2]
            - first_frame_idx: int
    
    Output:
        Dict mapping object_id -> List of N binary masks (H, W) as numpy uint8
        Masks are 0 (background) or 255 (object)
        If object not yet visible, mask is all zeros.
    
    Implementation:
    
    STEP 1: Setup SAM2 video predictor
        # Save frames as JPEGs to a temp directory (SAM2 requirement)
        # Initialize inference state
        inference_state = predictor.init_state(video_path=temp_frame_dir)
    
    STEP 2: Register all objects with their initial bboxes
        for det in initial_detections:
            frame_idx = det["first_frame_idx"]
            bbox = det["initial_bbox"]  # [x1, y1, x2, y2]
            obj_id = det["object_id"]
            
            # Add bbox prompt for this object on its first frame
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=np.array(bbox, dtype=np.float32)
            )
            
            # Initialize Kalman filter for this object
            self.kalman_filters[obj_id] = KalmanFilter(
                initial_bbox=bbox,
                process_noise=config.kalman_process_noise,
                measurement_noise=config.kalman_measurement_noise
            )
    
    STEP 3: Propagate with SAMURAI modifications
        # SAM2's propagate_in_video processes all frames sequentially
        # The image encoder runs ONCE per frame (shared across all objects)
        # The mask decoder runs per-object but is lightweight
        
        all_masks = {obj_id: [None] * len(frames) for obj_id in object_ids}
        
        for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state=inference_state
        ):
            # out_mask_logits shape: (num_objects, 1, H, W)
            
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                
                # === SAMURAI MODIFICATION 1: Kalman scoring ===
                # Get predicted bbox from Kalman filter
                predicted_bbox = self.kalman_filters[obj_id].predict()
                
                # Compute actual bbox from mask
                actual_bbox = mask_to_bbox(mask)
                
                # Motion score: how well does mask align with Kalman prediction?
                motion_score = compute_iou(predicted_bbox, actual_bbox)
                
                # Update Kalman filter with actual observation
                if motion_score > 0.1:  # Object visible
                    self.kalman_filters[obj_id].update(actual_bbox)
                
                # === SAMURAI MODIFICATION 2: Memory quality gating ===
                # Compute mask affinity score with stored good masks
                affinity_score = compute_mask_affinity(
                    mask, self.memory_banks[obj_id]
                )
                
                # Compute occurrence score from SAM2's occlusion head
                occurrence_score = torch.sigmoid(out_mask_logits[i, 0]).max().item()
                
                # Combined quality score
                quality = (
                    config.motion_score_weight * motion_score +
                    config.affinity_score_weight * affinity_score +
                    config.occurrence_score_weight * occurrence_score
                )
                
                # Only add to memory if quality is high enough
                if quality >= config.memory_quality_threshold:
                    self.memory_banks[obj_id].add(frame_idx, mask, quality)
                
                # Store the mask
                all_masks[obj_id][frame_idx] = mask
        
        return all_masks
    """
```

### File: pass3_tracking/kalman_filter.py

```
Class: KalmanFilter
    """
    Standard linear Kalman filter for bounding box tracking.
    
    State vector (8D): [cx, cy, w, h, vcx, vcy, vw, vh]
        cx, cy: center x, y of bounding box
        w, h: width, height of bounding box
        vcx, vcy: velocity of center
        vw, vh: velocity of width/height (for scale change)
    
    Constant velocity motion model:
        cx_new = cx + vcx
        cy_new = cy + vcy
        w_new = w + vw
        h_new = h + vh
    
    Measurement vector (4D): [cx, cy, w, h]
        Directly observed from the mask bounding box.
    
    Methods:
        __init__(initial_bbox, process_noise, measurement_noise)
            - Convert [x1, y1, x2, y2] to [cx, cy, w, h]
            - Initialize state and covariance matrices
            
        predict() -> [x1, y1, x2, y2]
            - Run Kalman predict step
            - Return predicted bbox in [x1, y1, x2, y2] format
            
        update(bbox: [x1, y1, x2, y2])
            - Convert bbox to [cx, cy, w, h]
            - Run Kalman update step
    
    Use numpy or filterpy library for the Kalman filter math.
    filterpy is simpler: pip install filterpy
    """
```

---

## Detailed Implementation — Main Pipeline

### File: pipeline.py

```python
"""
Main entry point. Orchestrates all 3 passes.

Usage:
    python pipeline.py --config config.yaml
    python pipeline.py --config config.yaml --input video.mp4
    python pipeline.py --config config.yaml --input frames_directory/
"""

# Pseudocode for main():

def main(config_path, input_override=None):
    # 1. Load config
    config = load_yaml(config_path)
    if input_override:
        config.video.input_path = input_override
    
    # 2. Load video frames
    print("Loading video...")
    frames = load_video(config.video.input_path, 
                        max_frames=config.video.max_frames,
                        resize=config.video.resize_short_edge)
    print(f"Loaded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # 3. Pass 1: Optical Flow
    print("\n=== Pass 1: Computing Optical Flow ===")
    timer.start("pass1")
    flow_estimator = RAFTFlowEstimator(config.pass1_optical_flow)
    flows = flow_estimator.compute_all_flows(frames)
    timer.stop("pass1")
    print(f"Computed {len(flows)} flow maps in {timer.elapsed('pass1'):.1f}s")
    
    # Optionally save flow visualizations
    if config.output.save_flow:
        save_flow_visualizations(flows, config.output.output_dir)
    
    # 4. Pass 2: Motion Analysis
    print("\n=== Pass 2: Detecting Moving Objects ===")
    timer.start("pass2")
    motion_detector = MotionDetector(config.pass2_motion_analysis)
    detections = motion_detector.detect_moving_objects(flows, frames)
    timer.stop("pass2")
    print(f"Found {len(detections)} moving objects in {timer.elapsed('pass2'):.1f}s")
    
    for det in detections:
        print(f"  Object {det['object_id']}: bbox={det['initial_bbox']}, "
              f"first_frame={det['first_frame_idx']}, confidence={det['confidence']:.2f}")
    
    if len(detections) == 0:
        print("No moving objects detected. Exiting.")
        return
    
    # 5. Pass 3: SAMURAI Tracking
    print("\n=== Pass 3: Tracking with SAMURAI ===")
    timer.start("pass3")
    tracker = SAMURAIMultiObjectTracker(config.pass3_tracking)
    all_masks = tracker.track_objects(frames, detections)
    timer.stop("pass3")
    print(f"Tracked {len(detections)} objects through {len(frames)} frames "
          f"in {timer.elapsed('pass3'):.1f}s")
    
    # 6. Save outputs
    print("\n=== Saving Results ===")
    if config.output.save_masks:
        save_masks(all_masks, config.output.output_dir)
    
    if config.output.save_visualization:
        save_visualization_video(
            frames, all_masks, detections, 
            config.output.output_dir,
            alpha=config.output.visualization_alpha
        )
    
    # Save metadata
    save_metadata(detections, timer.all_elapsed(), config.output.output_dir)
    
    # Print summary
    total = timer.elapsed("pass1") + timer.elapsed("pass2") + timer.elapsed("pass3")
    print(f"\n=== Done! Total time: {total:.1f}s for {len(frames)} frames ===")
    print(f"  Pass 1 (Flow):     {timer.elapsed('pass1'):.1f}s")
    print(f"  Pass 2 (Detect):   {timer.elapsed('pass2'):.1f}s")
    print(f"  Pass 3 (Track):    {timer.elapsed('pass3'):.1f}s")
    print(f"  Throughput:        {len(frames)/total:.1f} FPS")
```

---

## Key Implementation Notes

### 1. SAMURAI's shared encoder (CRITICAL for multi-object speed)

SAM2's `propagate_in_video()` already shares the image encoder across objects. When you register multiple objects via `add_new_points_or_box()` with different `obj_id` values, the video predictor internally:
- Runs the image encoder ONCE per frame
- Runs the mask decoder N times (once per object) using the same image features
- This is why 4 objects is NOT 4x slower — it's ~1.3x slower

**Do NOT create separate predictor instances per object. Use one predictor with multiple obj_ids.**

### 2. SAMURAI modifications to SAM2

SAMURAI does NOT retrain SAM2. It modifies the inference loop only:
- Adds Kalman filter prediction before mask selection
- Adds quality-aware memory gating after mask selection
- Both are pure Python/numpy — no model weight changes

The cleanest way to implement this is to:
1. Clone SAMURAI repo (which includes a modified SAM2)
2. Modify the `propagate_in_video` loop to add multi-object support with shared encoder
3. The Kalman filter and memory scoring run OUTSIDE the model — they just filter which masks/memories are kept

### 3. Frame format requirements

- SAM2 video predictor requires frames saved as JPEG files in a directory
- Frames must be named in sorted order (e.g., 00000.jpg, 00001.jpg, ...)
- Create a temp directory, save frames, point SAM2 at it, clean up after
- Use `tempfile.mkdtemp()` for the temp directory

### 4. RAFT batch processing

- RAFT processes frame PAIRS, not individual frames
- For N frames you get N-1 flow maps
- Batch size of 8 means 8 frame pairs processed simultaneously
- On A6000 with RAFT-Small at 480p: expect ~45-60 pairs/second
- Total for 199 pairs: ~3-4 seconds

### 5. Camera motion compensation

If the camera is static (surveillance, fixed tripod), skip this — set `camera_motion_compensation: false`. It adds complexity and compute for no benefit with a static camera.

If the camera is moving (handheld, drone, car-mounted), this is essential. Without it, the entire frame will appear to be "moving" and you'll detect the background as a moving object.

The RANSAC affine estimation approach works well for most camera motions (pan, tilt, zoom). For more extreme parallax (close objects vs far background), you'd need a homography, but affine is good enough for most cases.

### 6. Handling objects that start moving mid-video

Some objects may be stationary at frame 0 but start moving at frame 50. The pipeline handles this through `first_frame_idx` — each object's SAMURAI tracker is initialized at the frame where it first starts moving, not necessarily frame 0.

SAM2 supports adding new object prompts at any frame during propagation. Objects added later simply don't have masks for earlier frames (those remain as zero masks).

---

## Dependencies (requirements.txt)

```
torch>=2.3.1
torchvision>=0.18.1
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
scipy>=1.11.0
filterpy>=1.4.5          # Kalman filter
PyYAML>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0        # For visualization
imageio[ffmpeg]>=2.31.0  # For video I/O

# RAFT - install from source
# git clone https://github.com/princeton-vl/RAFT.git third_party/RAFT

# SAMURAI (modified SAM2) - install from source
# git clone https://github.com/yangchris11/samurai.git third_party/samurai
# cd third_party/samurai/sam2 && pip install -e .
```

---

## Setup Commands

```bash
# 1. Create environment
conda create -n motion_seg python=3.10 -y
conda activate motion_seg

# 2. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Clone dependencies
mkdir third_party
git clone https://github.com/princeton-vl/RAFT.git third_party/RAFT
git clone https://github.com/yangchris11/samurai.git third_party/samurai

# 4. Install SAM2 from SAMURAI
cd third_party/samurai/sam2
pip install -e .
cd ../../..

# 5. Download RAFT-Small weights
mkdir -p checkpoints
# Download raft-small.pth from RAFT releases
gdown https://drive.google.com/uc?id=1InqgGDMitIjfkNGZQPkDkeF3MCSxQMBi -O checkpoints/raft-small.pth

# 6. Download SAM2 Base+ checkpoint  
# SAM2 auto-downloads on first use, or manually:
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# 7. Install remaining dependencies
pip install -r requirements.txt

# 8. Run pipeline
python pipeline.py --config config.yaml --input my_video.mp4
```

---

## Expected Output

```
Loading video...
Loaded 200 frames at 854x480

=== Pass 1: Computing Optical Flow ===
Computed 199 flow maps in 3.2s

=== Pass 2: Detecting Moving Objects ===
Found 3 moving objects in 0.4s
  Object 0: bbox=[120, 80, 290, 340], first_frame=0, confidence=0.92
  Object 1: bbox=[450, 150, 580, 410], first_frame=0, confidence=0.85
  Object 2: bbox=[650, 200, 780, 380], first_frame=12, confidence=0.71

=== Pass 3: Tracking with SAMURAI ===
Tracked 3 objects through 200 frames in 11.8s

=== Done! Total time: 15.4s for 200 frames ===
  Pass 1 (Flow):     3.2s
  Pass 2 (Detect):   0.4s
  Pass 3 (Track):    11.8s
  Throughput:        13.0 FPS

Results saved to outputs/
  outputs/masks/object_0/ (200 PNG files)
  outputs/masks/object_1/ (200 PNG files)
  outputs/masks/object_2/ (200 PNG files)
  outputs/visualizations/overlay_video.mp4
  outputs/metadata.json
```

---

## Edge Cases to Handle

1. **No moving objects detected:** Print warning and exit gracefully
2. **Object leaves frame:** Kalman filter predicts exit, SAMURAI's occlusion head flags it, mask becomes empty
3. **Object re-enters frame:** Kalman filter predicts re-entry, SAMURAI re-acquires from memory
4. **Two objects merge/overlap:** Each has independent tracker and memory — they maintain separate IDs
5. **Very large objects (>50% of frame):** May be background parallax, not a real object — filter by max_object_area
6. **Static video (no motion):** Pass 2 returns empty list, pipeline exits with message
7. **Very short video (<10 frames):** Reduce temporal_window config to 2-3

---

## Testing

```bash
# Test with a sample video (use any video with moving objects)
# Davis 2016 dataset has good test videos: https://davischallenge.org/

# Quick test (first 50 frames only)
python pipeline.py --config config.yaml --input test_video.mp4

# Override max frames for quick testing
# Edit config.yaml: video.max_frames: 50
```
