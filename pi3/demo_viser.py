import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2

from pi3.pi3_inference import pi3_inference


def viser_wrapper(
    pred_dict: dict,
    port: int = 8081,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    background_mode: bool = False,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (V, 3, H, W)   - Input images,
                "points": (V, H, W, 3)   - 3D points in world coordinates,
                "conf": (V, H, W, 1)     - Confidence scores,
                "masks": (V, H, W)       - Boolean masks,
                "camera_poses": (V, 4, 4) - Camera-to-world transformation matrices,
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        background_mode (bool): Whether to run the server in background thread.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (V, 3, H, W)
    world_points = pred_dict["points"]  # (V, H, W, 3)
    conf = pred_dict["conf"]  # (V, H, W, 1)
    masks = pred_dict["masks"]  # (V, H, W)
    camera_poses = pred_dict.get("camera_poses", None)  # (V, 4, 4)

    # Remove the last dimension from conf if it exists
    if conf.ndim == 4:
        conf = conf[..., 0]  # (V, H, W)

    # Apply masks to confidence
    conf = conf * masks.astype(conf.dtype)

    # Convert images from (V, 3, H, W) to (V, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)  # now (V, H, W, 3)
    V, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center

    # Recenter camera poses if available
    if camera_poses is not None:
        cam_to_world = camera_poses[:, :3, :]  # (V, 3, 4)
        cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(V), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=-100, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(V)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="pi3_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (V, 3, 4)
        images_:    (V, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(V)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # Simple approximate FOV
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene if camera poses are available
    if camera_poses is not None:
        visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


parser = argparse.ArgumentParser(description="Pi3 demo with viser for 3D visualization")
parser.add_argument("--image_folder", type=str, default="examples/skating/", help="Path to folder containing images")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8081, help="Port number for the viser server")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--n_images", type=int, default=-1, help="Number of images to use for visualization")
parser.add_argument("--visualize_cache_file", type=str, default=None, help="Path to cached predictions")
parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization and only save predictions")


def main():
    """
    Main function for the Pi3 demo with viser for 3D visualization.

    This function:
    1. Loads the Pi3 model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points
    4. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --n_images: Number of images to use for visualization
    --visualize_cache_file: Path to cached predictions
    --skip_visualization: Skip visualization and only save predictions
    """
    args = parser.parse_args()

    if args.visualize_cache_file:
        predictions = torch.load(args.visualize_cache_file)
    else:
        # Determine precision
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        predictions = pi3_inference(image_folder=args.image_folder, n_images=args.n_images, precision=dtype)

    if not args.skip_visualization:
        print("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()  # convert to numpy

        print("Starting viser visualization...")

        viser_server = viser_wrapper(
            predictions,
            port=args.port,
            init_conf_threshold=args.conf_threshold,
            background_mode=args.background_mode,
        )
        print("Visualization complete")


if __name__ == "__main__":
    main()
