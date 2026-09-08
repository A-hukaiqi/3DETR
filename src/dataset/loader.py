"""
loader.py

Utility functions for loading SUN RGB-D data from disk.
"""

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------
# HELPER: construct paths for a single SUN RGB-D scene
# ---------------------------------------------------------------------
def get_scene_paths(scene_dir: Path) -> dict[str, Path]:
    """
    Construct the file paths for a single SUN RGB-D scene.

    Parameters:
        scene_dir : Path
            Path to one scene directory (e.g. img_0063).

    Returns:
        dict
            Dictionary containing all required file paths.
    """

    scene_name = scene_dir.name

    # take the first file in corresponding folder
    rgb = next((scene_dir / "image").iterdir())
    depth = next((scene_dir / "depth").iterdir())
    extrinsics = next((scene_dir / "extrinsics").iterdir())

    return {
        "scene": scene_dir.name,
        "rgb": rgb,
        "depth": depth,
        "intrinsics": scene_dir / "intrinsics.txt",
        "extrinsics": extrinsics,
    }

# ---------------------------------------------------------------------
# HELPER: take a .jpg RGB image and convert it to an RGB matrix
# ---------------------------------------------------------------------
def load_rgb(rgb_path: Path) -> np.ndarray:
    """
    Parameters:
        rgb_path: Path to the RGB image (.jpg)

    Returns:
        np.ndarray: RGB image of shape (H, W, 3), dtype uint8
    """

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if rgb is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    # OpenCV loads images as BGR
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    return rgb


# ---------------------------------------------------------------------
# HELPER: take a .png depth image and convert it to a depth matrix (meters)
# ---------------------------------------------------------------------
def load_depth(depth_path: Path) -> np.ndarray:
    """
    Parameters:
        depth_path: Path to the depth image (.png)

    Returns: 
        np.ndarray: Depth image of shape (H, W), dtype float32, in meters.
    """

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    # Decode SUN RGB-D depth encoding
    depth = depth.astype(np.uint16)
    depth = (depth >> 3) | (depth << 13)
    depth = depth.astype(np.float32) / 1000.0

    return depth


# ---------------------------------------------------------------------
# HELPER: load a 3×3 camera intrinsic matrix
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# HELPER: load a 3×3 camera intrinsic matrix
# ---------------------------------------------------------------------
def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    """
    Load the camera intrinsic matrix.

    Returns:
        intrinsics : ndarray
            3 × 3 camera intrinsic matrix.
    """

    intrinsics = np.loadtxt(intrinsics_path)

    if intrinsics.shape == (9,):
        intrinsics = intrinsics.reshape(3, 3)

    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Expected a 3×3 intrinsic matrix, got {intrinsics.shape}."
        )

    return intrinsics


# ---------------------------------------------------------------------
# HELPER: extract the gravity-alignment rotation matrix (Rtilt)
# ---------------------------------------------------------------------

def load_rtilt(extrinsics_path: Path) -> np.ndarray:
    """
    Parameters: 
        extrinsics_path: Path to extrinsics.txt

    Returns
        np.ndarray: 3×3 rotation matrix.
    """

    extrinsics = np.loadtxt(extrinsics_path)

    if extrinsics.shape != (3, 4):
        raise ValueError(
            f"Expected a 3x4 extrinsics matrix, got {extrinsics.shape}."
        )

    return extrinsics[:, :3]