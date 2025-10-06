"""Preprocessing utilities for point cloud reconstruction.

This module provides simple functions to resize and normalize input
frames before they are passed to the depth estimator and feature
extractor.  Preprocessing ensures consistent resolution and channel
ranges across different input devices.  All functions operate on
NumPy arrays in RGB order.

Functions
---------
resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray
    Resize an RGB frame to the given (height, width) using bilinear
    interpolation.
normalize_frame(frame: np.ndarray) -> np.ndarray
    Normalize an RGB frame to the range [0, 1] and convert to float32.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to the specified height and width.

    Parameters
    ----------
    frame: np.ndarray
        The input image as an (H, W, 3) array in RGB order.
    size: tuple of int
        The desired output size given as (height, width).

    Returns
    -------
    np.ndarray
        The resized image in RGB order.
    """
    height, width = size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize an RGB image to [0, 1] floating point representation.

    Parameters
    ----------
    frame: np.ndarray
        The input image as an (H, W, 3) array in RGB order with
        integer values in the range [0, 255].

    Returns
    -------
    np.ndarray
        The normalized image as float32 with values in [0, 1].
    """
    return (frame.astype(np.float32) / 255.0)
