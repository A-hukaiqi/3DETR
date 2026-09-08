import numpy as np

def visualize_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    window_name: str = "Point Cloud",
    point_size: float = 2.0,
):
    import open3d as o3d
    """
    Parameters

    points : ndarray, shape (N,3)
        XYZ coordinates of the point cloud.

    colors : ndarray, shape (N,3), optional
        RGB values in either
            [0,255] uint8
        or
            [0,1] float.

    window_name : str
        Title of the visualization window.

    point_size : float
        Size of rendered points.
    """

    # Sanity checks
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a NumPy array")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected points of shape (N,3), got {points.shape}"
        )

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optional colors
    if colors is not None:
        if colors.shape != points.shape:
            raise ValueError(
                "colors must have the same shape as points"
            )
        
        colors = colors.astype(np.float64)

        # Normalize if necessary
        if colors.max() > 1.0:
            colors /= 255.0

        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()


# define a helper function to display RGB or depth images
def display_image(image: np.ndarray,
                  title: str = "Image",
                  cmap: str | None = None) -> None:
    import matplotlib.pyplot as plt
    """
    Parameters

    image : np.ndarray
        Image to display.

    title : str
        Figure title.

    cmap : str | None
        Colormap for grayscale images.
    """

    # ---------- Type checking ----------
    if image is None:
        raise ValueError("Image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got {type(image)}"
        )

    # ---------- Print useful information ----------
    print(f"Type : {type(image)}")
    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")

    # ---------- Plot ----------
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()