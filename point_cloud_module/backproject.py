"""Back‑projection utilities.

This module contains functions to convert 2D pixel coordinates and
depth values into 3D camera coordinate points.  It supports
per‑pixel back‑projection using intrinsic parameters (focal lengths
fx, fy and principal point cx, cy).  Colour values or feature
embeddings can optionally be attached to each 3D point for
visualization or further processing.

Functions
---------
backproject(depth_map, intrinsics) -> np.ndarray
    Convert a depth map into an (N, 3) array of 3D points.
colourize_points(points, rgb) -> np.ndarray
    Attach RGB colour values to each point to form an (N, 6) array.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def backproject(depth_map: np.ndarray, intrinsics: Dict[str, float]) -> np.ndarray:
    """Back‑project a depth map into a cloud of 3D points.

    Parameters
    ----------
    depth_map: np.ndarray
        A (H, W) array of depth values (arbitrary units).  Zero
        or negative values are ignored.
    intrinsics: dict
        A dictionary with keys 'fx', 'fy', 'cx', 'cy'.  These
        parameters define the camera's focal length and principal
        point in pixel units.

    Returns
    -------
    np.ndarray
        A (N, 3) array of XYZ coordinates in the camera frame.
    """
    h, w = depth_map.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    zs = depth_map
    # Flatten
    xs_flat = xs.reshape(-1).astype(np.float32)
    ys_flat = ys.reshape(-1).astype(np.float32)
    zs_flat = zs.reshape(-1).astype(np.float32)
    # Filter out invalid depths
    valid_mask = zs_flat > 0
    xs_flat = xs_flat[valid_mask]
    ys_flat = ys_flat[valid_mask]
    zs_flat = zs_flat[valid_mask]
    # Back‑project
    X = (xs_flat - cx) * zs_flat / fx
    Y = (ys_flat - cy) * zs_flat / fy
    Z = zs_flat
    points = np.stack([X, Y, Z], axis=1)
    return points


def colourize_points(points: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Attach RGB colour to each 3D point.

    Parameters
    ----------
    points: np.ndarray
        An (N, 3) array of 3D coordinates.
    rgb: np.ndarray
        A (H, W, 3) image in RGB order.  Values are assumed to
        correspond to the same pixel grid used to generate ``points``.

    Returns
    -------
    np.ndarray
        An (N, 6) array where the last three columns contain RGB
        values normalized to [0, 1].
    """
    h, w, _ = rgb.shape
    # Flatten and normalize colours
    colours = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    # Use the same valid_mask logic as in backproject
    # Replicate a simple valid depth assumption: any non‑zero depth is valid
    # Here we assume the caller filtered points and colours accordingly.
    if points.shape[0] != colours.shape[0]:
        raise ValueError("Number of points and colours must match")
    return np.hstack([points, colours])
