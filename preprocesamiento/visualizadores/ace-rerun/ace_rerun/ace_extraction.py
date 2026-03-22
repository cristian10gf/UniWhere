"""ACE network point cloud extraction."""

import logging
import sys
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


def setup_ace_imports(ace_dir: Path):
    """Add the ACE source directory to sys.path."""
    ace_dir = ace_dir.resolve()
    if not ace_dir.exists():
        raise FileNotFoundError(f"ACE directory not found: {ace_dir}")
    if str(ace_dir) not in sys.path:
        sys.path.insert(0, str(ace_dir))


def extract_point_cloud_from_network(
    encoder_path: Path, head_path: Path, scene_dir: Path,
    ace_dir: Path, filter_depth: float, max_points: int, image_height: int,
):
    """
    Extract a colored 3D point cloud from a trained ACE network.
    Follows the approach described in ACE paper section B.5.
    """
    import torch
    from torch.amp import autocast
    from torch.utils.data import DataLoader

    setup_ace_imports(ace_dir)
    from ace_network import Regressor
    from ace_util import get_pixel_grid, to_homogeneous
    from dataset import CamLocDataset
    from skimage import color, io
    from skimage.transform import resize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_half = device.type == "cuda"

    _logger.info(f"Loading encoder: {encoder_path}")
    encoder_state = torch.load(encoder_path, map_location=device, weights_only=True)
    _logger.info(f"Loading head: {head_path}")
    head_state = torch.load(head_path, map_location=device, weights_only=True)

    network = Regressor.create_from_split_state_dict(encoder_state, head_state)
    network = network.to(device).eval()
    if use_half:
        network = network.half()

    train_dir = scene_dir / "train"
    dataset = CamLocDataset(root_dir=train_dir, mode=0, image_height=image_height)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    filter_repro_error = 1
    pixel_grid = get_pixel_grid(network.OUTPUT_SUBSAMPLE)
    avg_pc_points = max(1, max_points // len(loader))

    pc_xyz_list, pc_clr_list = [], []

    with torch.no_grad():
        for image, _, _, gt_inv_pose, K, _, _, file in loader:
            image = image.to(device, non_blocking=True)
            gt_inv_pose = gt_inv_pose.to(device, non_blocking=True)
            K = K.to(device, non_blocking=True)

            with autocast(device.type, enabled=use_half):
                scene_coords = network(image)

            B, C, H, W = scene_coords.shape
            pred_sc = scene_coords.float()
            pred_sc_4N = to_homogeneous(pred_sc.flatten(2))
            pred_cam = torch.matmul(gt_inv_pose[:, :3], pred_sc_4N)

            pred_px = torch.matmul(K, pred_cam)
            pred_px[:, 2].clamp_(min=0.1)
            pred_px_2N = pred_px[:, :2] / pred_px[:, 2, None]

            pix_pos = pixel_grid[:, :H, :W].clone().view(2, -1)
            repro_err = torch.norm(pred_px_2N.squeeze() - pix_pos.to(device), dim=0, keepdim=True, p=1)

            depth_mask = pred_cam[0, 2] < filter_depth
            err_mask = repro_err.squeeze() < filter_repro_error

            if err_mask.sum() < avg_pc_points:
                sorted_err, _ = torch.sort(repro_err.squeeze())
                err_mask = repro_err.squeeze() < sorted_err[min(avg_pc_points, sorted_err.shape[0] - 1)]

            vis_mask = torch.logical_and(depth_mask, err_mask)

            rgb = io.imread(file[0])
            if len(rgb.shape) < 3:
                rgb = color.gray2rgb(rgb)
            rgb = resize(rgb.astype("float64"), image.shape[2:])
            s = network.OUTPUT_SUBSAMPLE
            rgb = rgb[s // 2 :: s, s // 2 :: s, :]
            rgb = resize(rgb, scene_coords.shape[2:])
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().view(3, -1)

            rgb = rgb[:, vis_mask.cpu()]
            xyz = pred_sc_4N[0, :3, vis_mask].cpu()

            if xyz.shape[1] > avg_pc_points:
                stride = max(1, xyz.shape[1] // avg_pc_points)
                xyz, rgb = xyz[:, ::stride], rgb[:, ::stride]

            pc_xyz_list.append(xyz.numpy())
            pc_clr_list.append(rgb.numpy())

    positions = np.concatenate(pc_xyz_list, axis=1).T
    colors_float = np.concatenate(pc_clr_list, axis=1).T
    colors = (np.clip(colors_float, 0, 1) * 255).astype(np.uint8)

    _logger.info(f"Extracted {positions.shape[0]} points from ACE network")
    return positions, colors
