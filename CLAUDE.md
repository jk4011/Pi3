# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pi3** is a permutation-equivariant feed-forward network for 3D visual geometry reconstruction from unordered image sets. It predicts affine-invariant camera poses, scale-invariant local point maps, dense confidence scores, and metric scale — all without a fixed reference view.

**Pi3X** (recommended) is the enhanced version with convolutional output heads, multimodal conditioning (poses/intrinsics/depth), and metric scale support.

- Paper: https://arxiv.org/abs/2507.13347
- Upstream: https://github.com/yyfz/Pi3 (fork origin: https://github.com/jk4011/Pi3)
- HuggingFace models: `yyfz233/Pi3`, `yyfz233/Pi3X`
- Training code: `upstream/training` branch; Evaluation code: `upstream/evaluation` branch

## Setup & Commands

Conda environment: **`seg123`** (`conda activate seg123`)

```bash
pip install -r requirements.txt        # Core deps (torch/torchvision assumed pre-installed)
pip install -r requirements_demo.txt   # Gradio demo deps
pip install -e .                       # Editable install

# Inference
python example_mm.py                                    # Pi3X on default skating.mp4
python example_mm.py --data_path <dir_or_video>         # Custom data
python example_mm.py --data_path examples/room/rgb \
  --conditions_path examples/room/condition.npz \
  --save_path output.ply                                # With multimodal conditions

python example_vo.py --data_path <video>                # Video odometry (chunked)

python demo_gradio.py                                   # Gradio web UI
```

No test suite exists in this branch. Evaluation is on the `upstream/evaluation` branch.

## Architecture

### Pipeline Flow

```
Images (B,N,3,H,W) + Optional Conditions
  → DINOv2 ViT-L/14 Encoder (pi3/models/dinov2/)
  → Transformer Decoder with RoPE (layers/block.py, layers/attention.py)
  → Output Heads:
      - ConvHead → local point maps XY+Z (layers/conv_head.py)
      - CameraHead → 4x4 pose matrices via SVD orthogonalization (layers/camera_head.py)
      - ConvHead → confidence logits (sigmoid for [0,1])
      - Linear → metric scale factor
  → Global points = unproject(local_points, camera_poses)
```

### Key Model Files

| File | Role |
|------|------|
| `pi3/models/pi3x.py` | **Pi3X model** — multimodal conditioning, ConvHead outputs, metric scale |
| `pi3/models/pi3.py` | Original Pi3 model — simpler LinearPts3d head |
| `pi3/models/layers/block.py` | Transformer blocks: `BlockRope`, `PoseInjectBlock`, `CrossOnlyBlockRope` |
| `pi3/models/layers/attention.py` | `FlashAttentionRope`, `FlashCrossAttentionRope` (xFormers optional) |
| `pi3/models/layers/pos_embed.py` | `RoPE2D` (freq=100), `PositionGetter` |
| `pi3/models/layers/conv_head.py` | `ConvHead` — convolutional decoder with progressive upsampling |
| `pi3/models/layers/camera_head.py` | `CameraHead` — translation + SO(3) rotation (9D→SVD) |
| `pi3/models/layers/prope.py` | Projective Position Encoding for camera-aware attention |
| `pi3/pipe/pi3x_vo.py` | `Pi3XVO` — video odometry with SIM(3)/Umeyama chunk stitching |

### Data I/O (`pi3/utils/`)

- `basic.py`: `load_images_as_tensor()`, `load_multimodal_data()`, `write_ply()` — images auto-resized to multiples of 14, capped at 255k pixels
- `geometry.py`: `depth_edge()`, SE(3) operations, Umeyama alignment, ray computation

### Tensor Conventions

- **Images**: `(B, N, 3, H, W)` in `[0, 1]`
- **Poses**: `(B, N, 4, 4)` camera-to-world, OpenCV convention (right-down-forward)
- **Points**: `(B, N, H, W, 3)` — `local_points` per-view, `points` global
- **Confidence**: `(B, N, H, W, 1)` raw logits — apply `torch.sigmoid()` for probabilities
- **Conditions** (Pi3X): `poses`, `intrinsics (3x3)`, `depths (H,W)`, `rays (H,W,3)` + boolean masks

### Multimodal Conditioning (Pi3X)

Depth is encoded via a separate DINOv2-based encoder. Intrinsics/rays are injected via patch embedding. Poses are injected through `PoseInjectBlock` with attention masking. Each modality has a per-frame boolean mask (`mask_add_depth`, `mask_add_pose`, `mask_add_ray`) controlling which frames receive conditioning.

### Mixed Precision

Use `torch.bfloat16` on Ampere+ GPUs (compute capability ≥ 8), otherwise `torch.float16`.

## Common Post-processing Pattern

```python
masks = torch.sigmoid(results['conf'][..., 0]) > 0.1
non_edge = ~depth_edge(results['local_points'][..., 2], rtol=0.03)
masks = masks & non_edge
```
