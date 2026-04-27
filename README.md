# Motion-Aware Video Segmentation Pipeline

Automatically detects moving objects in video using optical flow (RAFT), then tracks and segments them with pixel-level masks using SAMURAI (motion-aware SAM2). No manual prompts needed — the system discovers what's moving and tracks it end-to-end.

**Target hardware:** NVIDIA A6000 (48 GB VRAM)  
**Target performance:** ~15 s for 200 frames, 3–4 objects

## Architecture

| Pass | What it does | Where it runs |
|------|-------------|---------------|
| **Pass 1** | RAFT-Small optical flow (batch GPU) | `pass1_optical_flow/` |
| **Pass 2** | Flow magnitude threshold → morphological cleanup → connected components → temporal consistency → top-K bboxes | `pass2_motion_analysis/` |
| **Pass 3** | SAMURAI (modified SAM2) multi-object tracking with shared image encoder, Kalman filter, quality-aware memory | `pass3_tracking/` |

## Quick Start

```bash
# 1. Create environment
conda create -n motion_seg python=3.10 -y
conda activate motion_seg

# 2. Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Run the setup script (clones SAMURAI, downloads SAM2 checkpoint, installs deps)
#    Pass 1 uses torchvision's RAFT — weights auto-download on first run (no Google Drive).
bash setup.sh

# 4. Run the pipeline
python pipeline.py --config config.yaml --input my_video.mp4
```

## Manual Setup

```bash
# Clone SAMURAI (Pass 1 uses torchvision's RAFT — no clone or manual weights needed)
mkdir -p third_party
git clone https://github.com/yangchris11/samurai.git third_party/samurai

# Install SAM2 from SAMURAI
cd third_party/samurai/sam2 && pip install -e . && cd ../../..

# Download SAM2 Base+ checkpoint
mkdir -p checkpoints
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Process a video file
python pipeline.py --config config.yaml --input video.mp4

# Process a directory of JPEG frames
python pipeline.py --config config.yaml --input frames/

# Quick test with limited frames (edit config.yaml → video.max_frames: 50)
python pipeline.py --config config.yaml --input video.mp4
```

## Configuration

All tunable parameters live in `config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video.resize_short_edge` | 480 | Resize frames for speed (null = original) |
| `pass1.batch_size` | 8 | Frame pairs per GPU batch |
| `pass2.flow_magnitude_threshold` | 2.0 | Pixels above this are "moving" |
| `pass2.max_objects` | 4 | Maximum tracked objects |
| `pass2.temporal_window` | 5 | Frames an object must appear to be kept |
| `pass2.camera_motion_compensation` | true | Subtract camera motion via RANSAC |
| `pass3.sam2_model_size` | base_plus | SAM2 model variant |

## Output

```
outputs/
├── masks/
│   ├── object_0/   (per-frame PNGs)
│   ├── object_1/
│   └── ...
├── visualizations/
│   └── overlay_video.mp4
└── metadata.json
```

## Key Design Decisions

- **One SAM2 predictor, multiple obj_ids** — the image encoder runs once per frame, shared across all objects. 4 objects is ~1.3× slower, not 4×.
- **Kalman filter + memory scoring are post-processing** — they run outside SAM2 on its mask outputs. No model weights are modified.
- **SAM2 requires JPEG frames on disk** — a temp directory is created and cleaned up automatically.
- **Pass 1 uses torchvision's RAFT** — `raft_small(weights=Raft_Small_Weights.DEFAULT)`; weights auto-download (no Google Drive). N frames produce N−1 flow maps.
- **Camera motion compensation** uses RANSAC affine estimation on sparse flow vectors. Disable for static cameras.
