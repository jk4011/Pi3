import os
import glob
import torch
import numpy as np
import builtins

from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_multimodal_data, load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from jhutil import cache_output, print_time
from easydict import EasyDict


device = "cuda" if torch.cuda.is_available() else "cpu"


class Pi3State:
    """Singleton to persist model state across module reloads (survives autoreload)."""

    def __new__(cls):
        if not hasattr(builtins, '_pi3_state_instance'):
            instance = super().__new__(cls)
            instance.model = None
            builtins._pi3_state_instance = instance
        return builtins._pi3_state_instance


_state = Pi3State()


def load_model():
    global _state
    if _state.model is None:
        print("Loading Pi3X model...")
        _state.model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        print(f"Model loaded on {device}")
    return _state.model


def unload_model():
    global _state
    if _state.model is not None:
        _state.model.cpu()
        del _state.model
        _state.model = None
        torch.cuda.empty_cache()
        print("Model unloaded")


def _register_feat_hook(model, feat_layer):
    captured = {}

    def hook_fn(module, input, output):
        captured['feat'] = output.detach()

    handle = model.decoder[feat_layer].register_forward_hook(hook_fn)
    return handle, captured


@cache_output(func_name="_pi3_inference", override=False)
def _pi3_inference(
    image_folder: str = None,
    image_names: tuple = None,
    interval: int = 1,
    precision: torch.dtype = torch.float32,
    conditions_path: str = None,
    feat_layer: int = None,
    pixel_limit: int = 255000,
) -> EasyDict:
    if _state.model is None:
        load_model()

    model = _state.model

    conditions = {}
    if image_folder is not None:
        conditions = dict(intrinsics=None, poses=None, depths=None)
        if conditions_path is not None and os.path.exists(conditions_path):
            print(f"Loading conditions from {conditions_path}...")
            data_npz = np.load(conditions_path, allow_pickle=True)
            if 'poses' in data_npz:
                conditions['poses'] = data_npz['poses']
            if 'depths' in data_npz:
                conditions['depths'] = data_npz['depths']
            if 'intrinsics' in data_npz:
                conditions['intrinsics'] = data_npz['intrinsics']

        imgs, conditions = load_multimodal_data(image_folder, conditions, interval=interval, device=device, PIXEL_LIMIT=pixel_limit)
    elif image_names is not None:
        imgs = load_images_as_tensor(list(image_names), interval=1, PIXEL_LIMIT=pixel_limit).to(device)
        imgs = imgs[None]
    else:
        raise ValueError("Either image_folder or image_names must be provided")

    if precision == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        print("bfloat16 not supported, falling back to float16")
        precision = torch.float16

    hook_handle = None
    captured = None
    if feat_layer is not None:
        hook_handle, captured = _register_feat_hook(model, feat_layer)

    print("Running Pi3X inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            res = model(imgs=imgs, **conditions)

    if hook_handle is not None:
        hook_handle.remove()

    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    B, N, _, H, W = imgs.shape
    patch_h, patch_w = H // 14, W // 14

    extrinsics = res['camera_poses'][0]
    local_points = res['local_points'][0]
    points = res['points'][0]
    depth = local_points[..., 2]

    rays_xy = local_points[..., :2] / local_points[..., 2:3]
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=imgs.device, dtype=torch.float32),
        torch.arange(W, device=imgs.device, dtype=torch.float32),
        indexing='ij',
    )
    ray_x_mean = rays_xy[..., 0].mean(dim=(1, 2))
    ray_y_mean = rays_xy[..., 1].mean(dim=(1, 2))
    u_mean, v_mean = u_coords.mean(), v_coords.mean()
    rx_centered = rays_xy[..., 0] - ray_x_mean[:, None, None]
    ry_centered = rays_xy[..., 1] - ray_y_mean[:, None, None]
    fx = ((u_coords[None] - u_mean) * rx_centered).mean(dim=(1, 2)) / (rx_centered ** 2).mean(dim=(1, 2))
    fy = ((v_coords[None] - v_mean) * ry_centered).mean(dim=(1, 2)) / (ry_centered ** 2).mean(dim=(1, 2))
    cx = u_mean - fx * ray_x_mean
    cy = v_mean - fy * ray_y_mean
    intrinsics = torch.zeros(N, 3, 3, device=imgs.device, dtype=torch.float32)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0

    predictions = EasyDict(
        world_points=points,
        local_points=local_points,
        depth=depth,
        conf=res['conf'][0, ..., 0],
        masks=masks,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        processed_images=imgs[0],
    )

    if captured is not None and 'feat' in captured:
        raw_feat = captured['feat']
        patch_start_idx = model.patch_start_idx
        hw = patch_h * patch_w + patch_start_idx

        if raw_feat.shape[0] == B and raw_feat.shape[1] == N * hw:
            raw_feat = raw_feat.reshape(B * N, hw, -1)

        feat_tokens = raw_feat[:, patch_start_idx:, :]
        predictions.feat = feat_tokens.reshape(N, patch_h, patch_w, -1)

    print("Inference completed")
    return predictions


def pi3_inference(
    image_folder: str = None,
    image_names: list = None,
    data_path: str = None,
    n_images: int = -1,
    interval: int = -1,
    precision: torch.dtype = torch.float32,
    conditions_path: str = None,
    feat_layer: int = None,
    upsample: bool = False,
    pca_dim: int = 3,
    pca_subsamples: int = 10000,
    depth_unprojection: bool = True,
    infer_gs: bool = False,
    camera_type: str = "c2w",
    pixel_limit: int = 255000,
) -> EasyDict:
    """
    Run Pi3X inference on images or video.

    Args:
        image_folder: Path to image directory or video file
        image_names: List of image file paths (overrides image_folder)
        data_path: Alias for image_folder (backward compat)
        n_images: Number of images to sample (-1 for all)
        interval: Frame sampling interval (-1 for auto: 1 for images, 10 for video)
        precision: torch.float16 or torch.bfloat16
        conditions_path: Path to .npz file with optional poses, depths, intrinsics
        feat_layer: Decoder layer index to extract features from (0-35)
        upsample: If True, upsample low-res features to high-res using AnyUp
        pca_dim: Target dimension after PCA for upsampling
        pca_subsamples: Number of samples for fitting PCA
        infer_gs: If True, infer Gaussian parameters
        depth_unprojection: If True, recompute world_points from depth + intrinsics + extrinsics
        camera_type: "c2w" (default, Pi3 native) or "w2c" (DA3 compat, N x 3 x 4)
    """
    if image_names is not None:
        image_names = [str(p) for p in image_names]

    if upsample:
        assert feat_layer is not None, "feat_layer must be provided if upsample is True"

    print("pi3 \\\n" +
        (f"--image_folder   {image_folder or data_path} \\\n" if (image_folder or data_path) is not None else "") +
        (f"--image_names    {' '.join(image_names)} \\\n" if image_names is not None else "") +
        (f"--conditions_path {conditions_path} \\\n"       if conditions_path is not None else "") +
        (f"--n_images       {n_images} \\\n"               if n_images != -1 else "") +
        (f"--interval       {interval} \\\n"               if interval != -1 else "") +
        (f"--feat_layer     {feat_layer} \\\n"             if feat_layer is not None else "") +
        (f"--upsample \\\n"                                if upsample else "") +
        (f"--pca_dim        {pca_dim} \\\n"                if upsample and pca_dim != 3 else "") +
        (f"--infer_gs \\\n"                                if infer_gs else "") +
        (f"--depth_unprojection \\\n"                      if depth_unprojection else "") +
        (f"--camera_type    {camera_type} \\\n"            if camera_type != "c2w" else "")
    )

    image_folder = image_folder or data_path

    if image_names is None and image_folder is None:
        raise ValueError("Either image_folder or image_names must be provided")

    if image_names is not None:
        if n_images > 0 and n_images < len(image_names):
            indices = np.linspace(0, len(image_names) - 1, n_images).astype(int)
            image_names = [image_names[i] for i in indices]

        print(f"Using {len(image_names)} images from list")
        prediction = _pi3_inference(
            image_names=tuple(image_names),
            precision=precision,
            feat_layer=feat_layer,
            pixel_limit=pixel_limit,
        )
    else:
        if interval < 0:
            interval = 10 if image_folder.lower().endswith('.mp4') else 1

        if n_images > 0 and os.path.isdir(image_folder):
            total = len([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if total > n_images:
                interval = max(1, total // n_images)

        print(f"Loading data from {image_folder} (interval={interval})...")
        prediction = _pi3_inference(
            image_folder=image_folder, interval=interval, precision=precision,
            conditions_path=conditions_path, feat_layer=feat_layer,
        )

    for k, v in prediction.items():
        if isinstance(v, torch.Tensor) and not v.is_cuda:
            prediction[k] = v.cuda()

    if depth_unprojection:
        from depth_anything_3.da3_inference import unproject_depth_to_points
        c2w = prediction.extrinsics
        R_w2c = c2w[:, :3, :3].transpose(1, 2)
        t_w2c = -torch.einsum('nij,nj->ni', R_w2c, c2w[:, :3, 3])
        extrinsics_w2c = torch.zeros_like(c2w)
        extrinsics_w2c[:, :3, :3] = R_w2c
        extrinsics_w2c[:, :3, 3] = t_w2c
        extrinsics_w2c[:, 3, 3] = 1.0
        prediction.world_points = unproject_depth_to_points(
            prediction.depth, extrinsics_w2c, prediction.intrinsics,
        )

    if infer_gs:
        from depth_anything_3.da3_inference import get_gaussians
        prediction.gaussians = get_gaussians(
            depths=prediction.local_points[..., 2],
            points=prediction.world_points,
            images=prediction.processed_images,
            conf=prediction.conf,
            scale_multiplier=1e-4,
            device=torch.device("cuda"),
        )

    if hasattr(prediction, 'feat') and prediction.feat is not None and upsample:
        from depth_anything_3.da3_inference import upsample_features

        with print_time("upsample_features"):
            prediction.feat_hr = upsample_features(
                prediction.processed_images,
                prediction.feat,
                pca_dim=pca_dim,
                pca_subsamples=pca_subsamples,
            )

    if camera_type == "w2c":
        c2w = prediction.extrinsics
        R_w2c = c2w[:, :3, :3].transpose(1, 2)
        t_w2c = -torch.einsum('nij,nj->ni', R_w2c, c2w[:, :3, 3])
        prediction.extrinsics = torch.cat([R_w2c, t_w2c.unsqueeze(-1)], dim=-1)

    return prediction


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pi3X inference")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="Path to image directory or video file")
    parser.add_argument("--image_names", nargs="+", default=None,
                        help="List of image paths")
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Path to .npz file with poses, depths, intrinsics")
    parser.add_argument("--save_path", type=str, default="output.ply",
                        help="Path to save output PLY file")
    parser.add_argument("--n_images", type=int, default=-1,
                        help="Number of images to sample")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Frame sampling interval (-1 for auto)")
    parser.add_argument("--feat_layer", type=int, default=None,
                        help="Decoder layer index to extract features from (0-35)")
    parser.add_argument("--upsample", action='store_true',
                        help="Upsample features using AnyUp")
    parser.add_argument("--pca_dim", type=int, default=3,
                        help="PCA dimension for feature reduction")
    parser.add_argument("--infer_gs", action='store_true',
                        help="Infer Gaussian parameters")
    parser.add_argument("--depth_unprojection", action='store_true',
                        help="Recompute world_points from depth + intrinsics + extrinsics")
    parser.add_argument("--camera_type", type=str, default='c2w', choices=['c2w', 'w2c'],
                        help="Extrinsics format: c2w (Pi3 native) or w2c (DA3 compat)")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on")

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    predictions = pi3_inference(
        image_folder=args.image_folder,
        image_names=args.image_names,
        n_images=args.n_images,
        interval=args.interval,
        precision=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        conditions_path=args.conditions_path,
        feat_layer=args.feat_layer,
        upsample=args.upsample,
        pca_dim=args.pca_dim,
        infer_gs=args.infer_gs,
        depth_unprojection=args.depth_unprojection,
        camera_type=args.camera_type,
    )

    if args.save_path:
        masks = predictions.masks
        points = predictions.points[masks].cpu()
        colors = predictions.images.permute(0, 2, 3, 1)[masks]
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        write_ply(points, colors, args.save_path)
        print(f"Saved to {args.save_path}")

    if hasattr(predictions, 'feat') and predictions.feat is not None:
        print(f"Features (layer {args.feat_layer}): {predictions.feat.shape}")
    if hasattr(predictions, 'feat_hr') and predictions.feat_hr is not None:
        print(f"Features HR: {predictions.feat_hr.shape}")

    print("Done!")
