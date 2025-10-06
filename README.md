# Camera‑based 3D Point Cloud Reconstruction

This repository contains a **modular Python framework** for reconstructing a 3D point cloud from **a single RGB camera** (or static image) with optional support for **DINOv3/TXT depth estimation**.  The design draws inspiration from recent research on combining learned depth, feature embeddings and back‑projection to produce high‑quality 3D clouds from sparse views.

## Background

Single‑image 3D reconstruction remains challenging, but several lines of research demonstrate that combining **monocular depth prediction**, **learned features** and **geometry priors** can enable plausible reconstructions:

- **DepthSplat** connects Gaussian splatting and depth estimation.  It leverages pre‑trained monocular depth features and multi‑view matching to produce 3D Gaussians from depth maps【949621598188621†L74-L82】.  The architecture unprojects depth into 3D Gaussian centres, demonstrating that depth cues help shape point‑cloud structures【949621598188621†L262-L272】.
- **LRM (Large Reconstruction Model)** shows that scaling transformer capacity and training on vast multi‑view datasets allows single‑image NeRF reconstruction【51799308392336†L16-L26】.  While powerful, it requires heavy compute and outputs radiance fields instead of explicit point clouds.
- **LAM3D** enriches image features by aligning them with tri‑plane representations derived from 3D point clouds.  This alignment reduces geometric distortions and improves mesh fidelity【80197465578975†L51-L66】, suggesting that point‑cloud priors help single‑view reconstruction.
- **Depth‑Regularized Gaussian Splatting** uses a dense monocular depth map (aligned via sparse SfM) to regularize 3D Gaussian optimization【952816532019651†L49-L60】.  Depth guidance mitigates floating artifacts and enforces geometric consistency.
- **Consistency Diffusion Models** explore diffusion training with 3D structural priors and 2D image priors, showing that such priors improve reconstruction consistency【894716512267128†L50-L66】.

These works motivate our pipeline: we use monocular depth as the primary cue for back‑projection, optionally enrich points with DINOv3/TXT features, and provide hooks for point‑cloud regularization (voxel downsampling and outlier removal).  Unlike heavy models such as LRM, our implementation is lightweight and intended for interactive use on a laptop.

## Pipeline Overview

The reconstruction pipeline has the following stages:

1. **Capture** – Acquire an RGB frame from the built‑in MacBook camera, an external USB camera or a static image (`ImageSource`).  Each camera returns an image and (optional) camera intrinsics.
2. **Preprocessing** – Resize and normalize the frame for the depth estimator and feature extractor.
3. **Depth Estimation** – Predict a per‑pixel depth map using the selected model.  Two back‑ends are provided:
   - `midas`/`dpt` using MiDaS/DPT (small and easy to run), implemented in `depth/midas_depth.py`.
   - `dinov3`/`dinov3_txt` (stub) for DINOv3 or dino.txt depth heads.  You must install the DINOv3 libraries and provide weights; see `depth/dinov3_depth.py`.
4. **Feature Extraction (optional)** – Extract dense embeddings with DINOv3/TXT for each pixel (`features/dinov3_features.py`).  This can be used for colouring or clustering points.
5. **Back‑projection** – Convert pixel coordinates and depth to 3D camera coordinates using intrinsic parameters.  RGB values are attached to each point.
6. **Post‑processing** – Remove isolated outliers and optionally downsample the point cloud using voxel grids (`postprocess.py`).
7. **Visualization & Export** – Save the cloud as a PLY file and optionally view it with Open3D.

## Installation

1. Clone this repository:

    git clone <your-repo-url> && cd <your-repo>

2. Install the required Python packages.  The basic pipeline needs only OpenCV, NumPy and Pillow.  For depth estimation and visualization you will need additional dependencies.  For example:

    pip install numpy opencv-python pillow

    # Depth models (choose one)
    pip install torch torchvision  # for MiDaS/DPT

    # For visualization
    pip install open3d  # optional

    # DINOv3/TXT support (optional; heavy)
    pip install dinov2  # and download pre-trained weights separately

3. (Optional) Download pre‑trained weights for DINOv3 or dino.txt depth heads.  See the official repositories for instructions.

## Usage

You can run the pipeline from Python or via the command‑line interface.

### Python API

To reconstruct a scene from the MacBook front camera using MiDaS depth:

    from point_cloud_module.reconstructor import PointCloudReconstructor

    recon = PointCloudReconstructor(
        input_mode="mac_front_cam",
        depth_model="midas",
        downsample_voxel_size=0.02,  # optional voxel size in metres
        remove_outlier_radius=0.05,   # optional outlier removal radius
    )
    points = recon.reconstruct_frame()
    recon.save_pointcloud("scene.ply", points)
    recon.visualize(points)

To use a USB camera instead:

    recon = PointCloudReconstructor(input_mode="usb_cam", camera_id=1, depth_model="midas")
    points = recon.reconstruct_frame()

To process a static image:

    recon = PointCloudReconstructor(input_mode="image_file", image_path="image.jpg", depth_model="midas")
    points = recon.reconstruct_frame()

### Command Line

Run the CLI from the repository root to capture from the MacBook camera and save the result:

    python -m point_cloud_module.cli --input_mode mac_front_cam --depth_model midas --output my_scene.ply --downsample_voxel_size 0.02 --remove_outlier_radius 0.05

For a USB camera:

    python -m point_cloud_module.cli --input_mode usb_cam --camera_id 1 --depth_model midas --output usb.ply

For an image file:

    python -m point_cloud_module.cli --input_mode image_file --image_path test.jpg --depth_model midas --output image.ply

## Extending the Pipeline

- **Add new depth models** by implementing a new estimator in `depth/` and registering it in `reconstructor.py`.
- **Integrate feature embeddings** by extending `Dinov3FeatureExtractor` to load real DINOv3 models.  You can then attach these embeddings to points for semantic segmentation.
- **Improve post‑processing** by using more sophisticated methods such as statistical outlier removal from Open3D or incorporating normal estimation.
- **Support Gaussian splatting** by storing covariance matrices per point and rendering them via a differentiable renderer; see DepthSplat【949621598188621†L74-L82】 for inspiration.

## Limitations and Future Work

This repository is a starting point.  It currently uses placeholder implementations for DINOv3/TXT depth and features, returning uniform depth or zero embeddings when the necessary weights are missing.  To fully exploit DINOv3/TXT you must install the official libraries and load pre‑trained checkpoints.  Similarly, camera intrinsics are estimated heuristically; calibrating your device will improve 3D accuracy.  Finally, absolute scale cannot be recovered from a single image without additional cues; our pipeline produces relative geometry which may need scaling to match real‑world units.

## References

- Xu et al., “DepthSplat: Connecting Gaussian Splatting and Depth,” CVPR 2025【949621598188621†L74-L82】【949621598188621†L262-L272】.
- Hong et al., “LRM: Large Reconstruction Model for Single Image to 3D,” arXiv 2023【51799308392336†L16-L26】.
- Cui et al., “LAM3D: Large Image‑Point‑Cloud Alignment Model,” arXiv 2024【80197465578975†L51-L66】.
- Chung et al., “Depth‑Regularized Optimization for 3D Gaussian Splatting,” arXiv 2023【952816532019651†L49-L60】.
- Jiang et al., “Consistency Diffusion Models for Single‑Image 3D Reconstruction with Priors,” arXiv 2025【894716512267128†L50-L66】.
