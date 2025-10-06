"""Post processing utilities for 3D point clouds.

This module contains helper functions to clean and downsample
point clouds produced by the back projection step.  These functions
operate purely on NumPy arrays and do not depend on external
libraries.  For more advanced filtering and smoothing you may
choose to use libraries such as Open3D or PCL.

Functions
---------
remove_outliers(points, radius, min_neighbors) -> np.ndarray
    Remove points that have fewer than ``min_neighbors`` within
    ``radius`` distance.  This helps eliminate isolated noise.
downsample(points, voxel_size) -> np.ndarray
    Downsample the point cloud by creating voxels of size
    ``voxel_size`` and averaging points within each voxel.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def remove_outliers(points: np.ndarray, radius: float = 0.05, min_neighbors: int = 3) -> np.ndarray:
    """Remove isolated points based on a simple radius search.

    Parameters
    ----------
    points: np.ndarray
        An (N, 3) or (N, 6) array of 3D points.  If more than three
        columns are present, only the first three are used for
        neighbor queries; the remaining columns are preserved in the
        output.
    radius: float
        The radius within which to search for neighbors.
    min_neighbors: int
        The minimum number of neighbors required to keep a point.

    Returns
    -------
    np.ndarray
        The filtered point cloud.
    """
    coords = points[:, :3]
    kept = np.ones(coords.shape[0], dtype=bool)
    # Naive O(N^2) neighbor counting; for large clouds consider using
    # a spatial index such as KDTree (scipy.spatial.cKDTree).
    for i in range(coords.shape[0]):
        dist2 = np.sum((coords - coords[i]) ** 2, axis=1)
        if np.sum(dist2 < radius ** 2) < min_neighbors:
            kept[i] = False
    return points[kept]


def downsample(points: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """Voxel grid downsampling.

    Parameters
    ----------
    points: np.ndarray
        An (N, M) array where the first three columns are XYZ.
    voxel_size: float
        The side length of each voxel in the same units as the point
        coordinates.  A larger voxel size results in fewer points.

    Returns
    -------
    np.ndarray
        The downsampled point cloud.
    """
    coords = points[:, :3]
    voxel_indices = np.floor(coords / voxel_size).astype(np.int32)
    # Use a dict to accumulate points by voxel
    voxel_dict = {}
    for idx, voxel in enumerate(map(tuple, voxel_indices)):
        if voxel not in voxel_dict:
            voxel_dict[voxel] = []
        voxel_dict[voxel].append(points[idx])
    downsampled = []
    for pts in voxel_dict.values():
        pts_arr = np.vstack(pts)
        downsampled.append(pts_arr.mean(axis=0))
    return np.vstack(downsampled)
