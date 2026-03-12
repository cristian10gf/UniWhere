"""
ACE Relocalizer — 6DoF camera pose estimation from query images.

Loads a pretrained ACE encoder + scene-specific head, runs inference,
and estimates the camera pose via OpenCV PnP+RANSAC.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from backend.relocalization.ace_network import Regressor
from backend.relocalization.pose_solver import solve_pose

_logger = logging.getLogger(__name__)

# Image normalization constants (from ACE training on 7scenes).
_IMAGE_MEAN = 0.4
_IMAGE_STD = 0.25
_DEFAULT_IMAGE_HEIGHT = 480


@dataclass
class RelocalizationResult:
    """Result of a relocalization query."""
    pose: np.ndarray              # 4×4 camera-to-world matrix
    translation: np.ndarray       # (3,) translation vector (from pose)
    rotation: np.ndarray          # 3×3 rotation matrix (from pose)
    inlier_count: int             # RANSAC inlier count
    success: bool                 # whether pose estimation succeeded


class ACERelocalizer:
    """
    Loads trained ACE models and estimates 6DoF camera poses from query images.

    Uses OpenCV PnP+RANSAC instead of dsacstar for portable, build-free inference.

    Parameters
    ----------
    encoder_path : str or Path
        Path to the pretrained encoder weights (.pt).
        Default location: ``preprocesamiento/models/ace/ace_encoder_pretrained.pt``
    head_path : str or Path
        Path to the scene-specific head weights (.pt).
        Produced by the preprocessing pipeline at ``preprocesamiento/data/output/<scene>.pt``
    device : str or torch.device, optional
        Device for inference. Auto-detects CUDA if available.
    image_height : int
        Target height for input images (default: 480).
    """

    def __init__(
        self,
        encoder_path: Union[str, Path],
        head_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        image_height: int = _DEFAULT_IMAGE_HEIGHT,
    ):
        self.image_height = image_height

        # Auto-detect device.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._use_half = self.device.type == "cuda"

        # Load encoder and head state dicts.
        encoder_state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
        _logger.info(f"Loaded encoder from: {encoder_path}")

        head_state_dict = torch.load(head_path, map_location="cpu", weights_only=True)
        _logger.info(f"Loaded head from: {head_path}")

        # Build the regressor from split state dicts.
        self._network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
        self._network = self._network.to(self.device)
        self._network.eval()

        if self._use_half:
            self._network = self._network.half()

        _logger.info(f"ACERelocalizer ready on {self.device} "
                     f"(half={self._use_half}, target_height={self.image_height})")

    def relocalize(
        self,
        image: np.ndarray,
        focal_length: float,
        principal_point: Optional[tuple[float, float]] = None,
        reprojection_threshold: float = 10.0,
        ransac_iterations: int = 64,
    ) -> RelocalizationResult:
        """
        Estimate the 6DoF camera pose for a query image.

        Parameters
        ----------
        image : np.ndarray
            Input image as BGR (H, W, 3) or grayscale (H, W). uint8.
        focal_length : float
            Camera focal length in pixels at the original image resolution.
        principal_point : tuple of (cx, cy), optional
            Principal point in pixels at the original resolution.
            Defaults to the image center.
        reprojection_threshold : float
            RANSAC inlier threshold in pixels (default: 10.0).
        ransac_iterations : int
            Number of RANSAC iterations (default: 64).

        Returns
        -------
        RelocalizationResult
            Pose, translation, rotation, inlier count, and success flag.
        """
        # --- Preprocessing ---
        # Convert to grayscale if needed.
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        orig_h, orig_w = gray.shape[:2]

        # Default principal point = image center.
        if principal_point is None:
            pp_x = orig_w / 2.0
            pp_y = orig_h / 2.0
        else:
            pp_x, pp_y = principal_point

        # Resize to target height, preserving aspect ratio.
        scale_factor = self.image_height / orig_h
        new_w = int(orig_w * scale_factor)
        resized = cv2.resize(gray, (new_w, self.image_height), interpolation=cv2.INTER_AREA)

        # Scale intrinsics proportionally.
        scaled_focal = focal_length * scale_factor
        scaled_pp_x = pp_x * scale_factor
        scaled_pp_y = pp_y * scale_factor

        # Normalize: float [0,1] → (x - mean) / std.
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = (tensor - _IMAGE_MEAN) / _IMAGE_STD
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        if self._use_half:
            tensor = tensor.half()

        tensor = tensor.to(self.device, non_blocking=True)

        # --- Inference ---
        with torch.no_grad():
            with autocast(enabled=self._use_half):
                scene_coords = self._network(tensor)  # (1, 3, H/8, W/8)

        # Move to CPU float for PnP.
        scene_coords_3HW = scene_coords[0].float().cpu()

        # --- Pose estimation ---
        result = solve_pose(
            scene_coordinates_3HW=scene_coords_3HW,
            focal_length=scaled_focal,
            pp_x=scaled_pp_x,
            pp_y=scaled_pp_y,
            reprojection_threshold=reprojection_threshold,
            iterations=ransac_iterations,
            output_subsample=Regressor.OUTPUT_SUBSAMPLE,
        )

        pose = result.pose_4x4
        return RelocalizationResult(
            pose=pose,
            translation=pose[:3, 3].copy(),
            rotation=pose[:3, :3].copy(),
            inlier_count=result.inlier_count,
            success=result.success,
        )
