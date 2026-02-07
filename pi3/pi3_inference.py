import os
import glob
import torch
from typing import List, Optional

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from jhutil import cache_output

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


def load_model():
    """Load Pi3 model. If model is already loaded, return it."""
    global model
    if model is None:
        print(f"Loading Pi3 model...")
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        print(f"Model loaded on {device}")
    return model


def unload_model():
    """Unload model from memory."""
    global model
    if model is not None:
        model.cpu()
        del model
        model = None
        torch.cuda.empty_cache()
        print("Model unloaded")


@cache_output(func_name="_pi3_inference", override=False)
def _pi3_inference(
    image_names: list = None,
    interval: int = 1,
    precision: torch.dtype = torch.float16
) -> dict:
    """
    Cached Pi3 inference function.
    
    Args:
        image_names: List of image file paths
        interval: Interval to sample images (used when loading from video)
        precision: Precision for inference (torch.float16 or torch.bfloat16)
    
    Returns:
        Dictionary containing:
            - points: 3D points (N, V, H, W, 3)
            - local_points: Local 3D points (N, V, H, W, 3)
            - conf: Confidence scores (N, V, H, W, 1)
            - masks: Boolean masks (V, H, W)
            - images: Input images (V, 3, H, W)
    """

    # Load model if not already loaded
    if model is None:
        load_model()
        
    # Load images
    # For image list, we need to handle loading differently
    # Since load_images_as_tensor expects a path or video file
    # We'll need to load images one by one
    print(f"Loading {len(image_names)} images...")
    
    # Load images as tensor
    imgs = load_images_as_tensor(image_names[0] if len(image_names) == 1 else image_names, interval=interval).to(device)
    print(f"Loaded images shape: {imgs.shape}")
    
    # Determine precision
    if precision == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        print("bfloat16 not supported on this GPU, falling back to float16")
        precision = torch.float16
    
    # Run inference
    print("Running Pi3 inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            res = model(imgs[None])  # Add batch dimension
    
    # Process masks
    print("Processing masks...")
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]
    
    # Prepare output
    predictions = {
        'points': res['points'][0],  # (V, H, W, 3)
        'local_points': res['local_points'][0],  # (V, H, W, 3)
        'conf': res['conf'][0],  # (V, H, W, 1)
        'masks': masks,  # (V, H, W)
        'images': imgs,  # (V, 3, H, W)
        'camera_poses': res['camera_poses'][0],  # (V, 4, 4)
    }
    
    print("Inference completed")
    return predictions


def pi3_inference(
    image_folder: str = None,
    image_names: list = None,
    n_images: int = -1,
    precision: torch.dtype = torch.float16
) -> dict:
    """
    Run Pi3 inference on images.

    Args:
        image_folder: Path to image directory
        image_names: List of image file paths (overrides image_folder)
        n_images: Number of images to sample from the sequence (-1 for all)
        precision: Precision for inference (torch.float16 or torch.bfloat16)

    Returns:
        Dictionary containing inference results
    """

    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    if image_names is None:
        image_names = glob.glob(os.path.join(image_folder, "*"))
        try:
            image_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except:
            image_names.sort(key=lambda p: os.path.splitext(p)[0])

    if n_images > 0 and n_images < len(image_names):
        import numpy as np
        image_indices = np.linspace(0, len(image_names) - 1, n_images).astype(int)
        image_names = [image_names[i] for i in image_indices]

    print(f"Found {len(image_names)} images")

    # Run cached inference with interval=1 (default for images)
    return _pi3_inference(image_names, interval=1, precision=precision)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run Pi3 inference")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to image directory")
    parser.add_argument("--save_path", type=str, default="output.ply",
                        help="Path to save output PLY file")
    parser.add_argument("--n_images", type=int, default=-1,
                        help="Number of images to sample")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on")

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"

    # Run inference
    predictions = pi3_inference(
        image_folder=args.image_folder,
        n_images=args.n_images
    )

    print("Done!")


