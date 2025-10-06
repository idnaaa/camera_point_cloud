"""High‑level point cloud reconstruction pipeline.

The :class:`PointCloudReconstructor` orchestrates the entire process
from capturing a frame through depth estimation, feature extraction,
back‑projection and post‑processing to producing a 3D point cloud.
It exposes a simple interface for users and hides the complexity of
each module.

Supported input modes:
- ``"mac_front_cam"``: use the built‑in MacBook camera.
- ``"usb_cam"``: use an external USB camera (specify ``camera_id``).
- ``"image_file"``: load a frame from an image path (provide
  ``image_path``).

Supported depth models:
- ``"dinov3_txt"``: use the DINOv3/dino.txt depth head (requires
  additional dependencies).
- ``"midas"``: use the MiDaS/DPT family.

Usage example:

    recon = PointCloudReconstructor(
        input_mode="mac_front_cam",
        camera_id=0,
        depth_model="midas",
        output_format="ply",
    )
    cloud = recon.reconstruct_frame()
    recon.save_pointcloud("cloud.ply", cloud)
    recon.visualize(cloud)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .camera import MacFrontCam, UsbCam, ImageSource
from .preprocessing import resize_frame, normalize_frame
from .depth import Dinov3DepthEstimator, MidasDepthEstimator
from .features import Dinov3FeatureExtractor
from .backproject import backproject, colourize_points
from .postprocess import remove_outliers, downsample
from .visualize import visualize_point_cloud


@dataclass
class PointCloudReconstructor:
    """End‑to‑end point cloud reconstruction pipeline."""

    input_mode: str = "mac_front_cam"
    camera_id: int = 0
    image_path: Optional[str] = None
    depth_model: str = "midas"
    output_format: str = "ply"
    resize: Optional[tuple[int, int]] = (480, 640)
    downsample_voxel_size: Optional[float] = None
    remove_outlier_radius: Optional[float] = None
    remove_outlier_min_neighbors: int = 3
    # Instances of the submodules; initialised lazily
    _camera: Optional[object] = field(init=False, default=None)
    _depth_estimator: Optional[object] = field(init=False, default=None)
    _feature_extractor: Optional[object] = field(init=False, default=None)

    def _get_camera(self):
        if self._camera is not None:
            return self._camera
        if self.input_mode == "mac_front_cam":
            self._camera = MacFrontCam(camera_id=self.camera_id)
        elif self.input_mode == "usb_cam":
            self._camera = UsbCam(camera_id=self.camera_id)
        elif self.input_mode == "image_file":
            if not self.image_path:
                raise ValueError("image_path must be provided for image_file mode")
            self._camera = ImageSource(image_path=self.image_path)
        else:
            raise ValueError(f"Unknown input_mode {self.input_mode}")
        return self._camera

    def _get_depth_estimator(self):
        if self._depth_estimator is not None:
            return self._depth_estimator
        if self.depth_model in ("dinov3", "dinov3_txt"):
            self._depth_estimator = Dinov3DepthEstimator()
        elif self.depth_model in ("midas", "dpt"):
            self._depth_estimator = MidasDepthEstimator()
        else:
            raise ValueError(f"Unknown depth_model {self.depth_model}")
        return self._depth_estimator

    def _get_feature_extractor(self):
        if self._feature_extractor is not None:
            return self._feature_extractor
        # Only instantiate if the user selects a dinov3 feature model
        self._feature_extractor = Dinov3FeatureExtractor()
        return self._feature_extractor

    def reconstruct_frame(self) -> np.ndarray:
        """Capture a frame, estimate depth and return a 3D point cloud.

        Returns
        -------
        np.ndarray
            An (N, 6) or (N, 3) array of 3D points with optional RGB.
        """
        cam = self._get_camera()
        frame, intrinsics = cam.get_frame()
        # Preprocess
        if self.resize:
            frame = resize_frame(frame, self.resize)
        frame_norm = normalize_frame(frame)
        # Depth estimation
        depth_estimator = self._get_depth_estimator()
        depth_map = depth_estimator.predict(frame_norm)
        # Back‑project
        points = backproject(depth_map, intrinsics)
        # Attach colours
        # Flatten RGB corresponding to valid depths; simple approach
        # Compute mask of valid depths and apply to colours
        valid_mask = depth_map.reshape(-1) > 0
        colours = frame.reshape(-1, 3)[valid_mask]
        points_coloured = np.hstack([points, colours.astype(np.float32)])
        # Downsample if requested
        if self.downsample_voxel_size is not None:
            points_coloured = downsample(points_coloured, voxel_size=self.downsample_voxel_size)
        # Remove outliers if requested
        if self.remove_outlier_radius is not None:
            points_coloured = remove_outliers(
                points_coloured,
                radius=self.remove_outlier_radius,
                min_neighbors=self.remove_outlier_min_neighbors,
            )
        return points_coloured

    def save_pointcloud(self, filename: str, points: np.ndarray) -> None:
        """Save the point cloud to disk in a simple PLY format."""
        # Convert to standard PLY format
        with open(filename, "w") as f:
            num_points = points.shape[0]
            has_color = points.shape[1] == 6
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            for pt in points:
                if has_color:
                    x, y, z, r, g, b = pt
                    f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
                else:
                    x, y, z = pt
                    f.write(f"{x} {y} {z}\n")

    def visualize(self, points: np.ndarray) -> None:
        """Display the point cloud using Open3D (if available)."""
        visualize_point_cloud(points)
