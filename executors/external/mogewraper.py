import os
import sys

# You can keep or remove these paths additions if needed in your environment
paths = [
    r'E:\stable-diffusion\stable-diffusion-integrator',
    r'E:\track_anything_project',
    r'E:\ai_projects\dust3r_project\vision-forge',
    r'E:\nuke-bridge',
    'E:/ai_projects/ai_portal',
    r'E:\ai_projects\dust3r_project\MoGo',
    r'E:\ai_projects\dust3r_project\MoGo\MoGe'
]
for p in paths:
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import torch
import trimesh
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import Optional

# From your original code/repo
from MoGe import utils3d
from MoGe.moge.model import MoGeModel
from nukebridge.utils.image_io import get_image_io

# Enable EXR support in OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


@dataclass
class MoGeOutput:
    """
    Data class to store the output of the MoGe model inference.
    """
    points: np.ndarray  # Shape: (H, W, 3)
    depth: np.ndarray  # Shape: (H, W)
    mask: Optional[np.ndarray]  # Shape: (H, W)
    intrinsics: np.ndarray  # Shape: (3, 3)
    image: np.ndarray  # Original input image (H, W, 3)
    root_folder: Path  # Root folder for saving outputs

    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 24.0
    shift: float = 0

    imageIO = get_image_io()

    def estimate_focal_length_from_frustum(self):
        """
        Estimates the focal length based on the frustum dimensions.
        Returns:
        - focal_length_x_px: Estimated focal length (X) in pixels
        - focal_length_y_px: Estimated focal length (Y) in pixels
        """
        # Get image dimensions
        image_height, image_width = self.image.shape[:2]

        # Sensor dimensions
        sensor_width_mm = self.sensor_width_mm
        sensor_height_mm = self.sensor_height_mm

        # Get near face
        _, near_face_vertices = self.get_camera_frustum_faces()

        # Width/height of the near face
        W_near = np.linalg.norm(near_face_vertices[0, :2] - near_face_vertices[3, :2])
        H_near = np.linalg.norm(near_face_vertices[0, :2] - near_face_vertices[1, :2])

        # Distance from camera to near plane
        z_near = near_face_vertices[0, 2] + self.shift

        # Field of view angles
        theta_x = 2 * np.arctan((W_near / 2) / z_near)
        theta_y = 2 * np.arctan((H_near / 2) / z_near)

        # Focal lengths in mm
        focal_length_x_mm = (sensor_width_mm / 2) / np.tan(theta_x / 2)
        focal_length_y_mm = (sensor_height_mm / 2) / np.tan(theta_y / 2)

        # Convert mm -> px
        focal_length_x_px = focal_length_x_mm * (image_width / sensor_width_mm)
        focal_length_y_px = focal_length_y_mm * (image_height / sensor_height_mm)

        print(f"Estimated focal lengths: fx = {focal_length_x_mm:.2f} mm, fy = {focal_length_y_mm:.2f} mm")
        print(f"Estimated focal lengths: fx = {focal_length_x_px:.2f} px, fy = {focal_length_y_px:.2f} px")

        return focal_length_x_px, focal_length_y_px

    def get_face_at_depth(self, depth_percentage=100.0, depth_tolerance=5.0):
        """
        Computes a face in XY plane at the specified depth percentage,
        with some tolerance to create a slice.
        """
        if not (0 <= depth_percentage <= 100):
            raise ValueError("depth_percentage must be between 0 and 100.")

        if not (0 <= depth_tolerance <= 100):
            raise ValueError("depth_tolerance must be between 0 and 100.")

        # Min/max Z
        z_min = np.min(self.points[:, :, 2])
        z_max = np.max(self.points[:, :, 2])

        # Depth Z at specified percentage
        depth_z = z_min + (depth_percentage / 100.0) * (z_max - z_min)

        # Depth tolerance
        delta_z = (depth_tolerance / 100.0) * (z_max - z_min)
        depth_min = depth_z - delta_z
        depth_max = depth_z + delta_z

        # Select points in that Z range
        points_flat = self.points.reshape(-1, 3)
        within_depth = (points_flat[:, 2] >= depth_min) & (points_flat[:, 2] <= depth_max)
        depth_points = points_flat[within_depth]
        if depth_points.size == 0:
            raise ValueError("No points found within the specified depth range.")

        # Compute bounding box in X/Y
        max_abs_x = np.max(np.abs(depth_points[:, 0]))
        max_abs_y = np.max(np.abs(depth_points[:, 1]))

        # Construct a face
        face_vertices = np.array([
            [max_abs_x, max_abs_y, depth_z],
            [max_abs_x, -max_abs_y, depth_z],
            [-max_abs_x, -max_abs_y, depth_z],
            [-max_abs_x, max_abs_y, depth_z],
        ])
        return face_vertices

    def get_camera_frustum_faces(self, step=2, depth_tolerance=0.5):
        """
        Generates near/far faces of the frustum based on the point cloud.
        """
        abs_z = np.abs(self.points[:, :, 2])
        z_min, z_max = np.min(abs_z), np.max(abs_z)
        scaled_vertices_list = []

        for depth_percentage in range(0, 101, step):
            try:
                face_vertices = self.get_face_at_depth(depth_percentage, depth_tolerance)
                plan_depth = np.abs(face_vertices[0, 2])
                scale_factor = z_max / plan_depth if plan_depth != 0 else 1.0

                scaled_vertices = face_vertices.copy()
                scaled_vertices[:, 0] *= scale_factor
                scaled_vertices[:, 1] *= scale_factor
                scaled_vertices_list.append(scaled_vertices)
            except ValueError:
                continue

        if not scaled_vertices_list:
            raise ValueError("No valid faces were generated for the frustum.")

        all_scaled_vertices = np.vstack(scaled_vertices_list)
        max_abs_x = np.max(np.abs(all_scaled_vertices[:, 0]))
        max_abs_y = np.max(np.abs(all_scaled_vertices[:, 1]))

        z_far = z_max
        far_face_vertices = np.array([
            [max_abs_x, max_abs_y, z_far],
            [max_abs_x, -max_abs_y, z_far],
            [-max_abs_x, -max_abs_y, z_far],
            [-max_abs_x, max_abs_y, z_far],
        ])

        # Near face
        z_near = z_min if z_min != 0 else 0.001
        scale_ratio = z_near / z_far
        near_face_vertices = far_face_vertices.copy()
        near_face_vertices[:, 0] *= scale_ratio
        near_face_vertices[:, 1] *= scale_ratio
        near_face_vertices[:, 2] = z_near

        return far_face_vertices, near_face_vertices

    def save_frustum_as_obj(self, output_path):
        """
        Generates camera frustum mesh and saves it as OBJ.
        """
        far_face_vertices, near_face_vertices = self.get_camera_frustum_faces()
        vertices = np.vstack((near_face_vertices, far_face_vertices))

        # Flip Z
        vertices[:, 2] *= -1

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Near face
            [4, 5, 6], [4, 6, 7],  # Far face
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ])
        faces = faces[:, ::-1]
        frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        output_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        output_path = output_path.with_suffix('.obj')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frustum_mesh.export(str(output_path))
        print(f"Frustum saved to {output_path}")

    def get_camera_parameters(self):
        """
        Compute camera parameters in mm, position, rotation.
        """
        focal_length_x_px, focal_length_y_px = self.estimate_focal_length_from_frustum()
        image_height, image_width = self.image.shape[:2]
        c_x = image_width / 2
        c_y = image_height / 2

        # Update intrinsics
        self.intrinsics = np.array([
            [focal_length_x_px, 0, c_x],
            [0, focal_length_y_px, c_y],
            [0, 0, 1]
        ])

        # Convert px -> mm
        focal_length_x_mm = focal_length_x_px * (self.sensor_width_mm / image_width)
        focal_length_y_mm = focal_length_y_px * (self.sensor_height_mm / image_height)

        position = np.array([0.0, 0.0, 0.0])
        rotation = np.array([0.0, 0.0, 0.0])

        camera_params = {
            'focal_length_x_mm': focal_length_x_mm,
            'focal_length_y_mm': focal_length_y_mm,
            'position': position,
            'rotation': rotation,
            'sensor_width_mm': self.sensor_width_mm,
            'sensor_height_mm': self.sensor_height_mm
        }
        return camera_params

    def get_camera_data(self):
        """
        Extract focal lengths & FOV from intrinsics.
        """
        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(self.intrinsics)
        focal_length_x = self.intrinsics[0, 0]
        focal_length_y = self.intrinsics[1, 1]

        return {
            'fov_x': np.rad2deg(fov_x),
            'fov_y': np.rad2deg(fov_y),
            'focal_length_x': focal_length_x,
            'focal_length_y': focal_length_y
        }

    def save_mesh(self, output_path, file_format='ply', remove_edge=True, rtol=0.02):
        """
        Create a mesh from points, optionally remove edges, and save.
        """
        points = self.points
        image = self.image
        mask = self.mask

        height, width = image.shape[:2]

        # Resolve output path
        out_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        out_path = out_path.with_suffix(f'.{file_format}')

        if mask is None:
            mask = np.ones((height, width), dtype=bool)

        # Remove edges if requested
        if remove_edge:
            depth = points[..., 2]
            edge_mask = ~utils3d.numpy.depth_edge(depth, mask=mask, rtol=rtol)
            mask = mask & edge_mask

        # Build mesh
        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            image.astype(np.float32) / 255.0,
            utils3d.numpy.image_uv(width=width, height=height),
            mask=mask,
            tri=True
        )

        # Flip coordinates for standard axes & UV
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

        # Trimesh object
        if file_format.lower() == 'ply':
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False
            )
        else:
            # e.g. OBJ with texture
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=trimesh.visual.TextureVisuals(
                    uv=vertex_uvs,
                    image=Image.fromarray(image)
                ),
                process=False
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_path))

    def save_depth_map(self, output_path):
        """
        Save depth map as EXR.
        """
        out_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        exr_path = str(out_path.with_suffix('.exr'))
        self.imageIO.write_image(self.depth, exr_path, image_format='exr')

    def save_point_cloud(self, output_path):
        """
        Save the point cloud as a PLY file.
        """
        points = self.points
        mask = self.mask

        out_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        out_path = out_path.with_suffix('.ply')

        valid_points = points[mask] if mask is not None else points.reshape(-1, 3)
        pc = trimesh.PointCloud(valid_points)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pc.export(str(out_path))

    def save_camera_nuke(self, output_path):
        """
        Save camera parameters as a Nuke .nk file.
        """
        cam_params = self.get_camera_parameters()
        focal_length_mm = cam_params['focal_length_x_mm']
        haperture_mm = cam_params['sensor_width_mm']
        vaperture_mm = cam_params['sensor_height_mm']
        position = cam_params['position']
        rotation = cam_params['rotation']

        camera_nk_content = f"""
Camera2 {{
 inputs 0
 name Camera1
 focal {focal_length_mm}
 haperture {haperture_mm}
 vaperture {vaperture_mm}
 translate {{ {position[0]} {position[1]} {position[2]} }}
 rotate {{ {rotation[0]} {rotation[1]} {rotation[2]} }}
}}
"""
        out_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        out_path = out_path.with_suffix('.nk')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(camera_nk_content)


class MoGeInference:
    def __init__(
            self,
            model_path: str,
            root_folder: str = '.',
            device: str = 'cuda',
            sensor_width_mm: float = 36.0,
            sensor_height_mm: float = 24.0
    ):
        """
        Initialize the MoGeInference class.

        Parameters:
        - model_path (str or Path): Path or name of the pre-trained MoGe model.
        - root_folder (str or Path): Root folder for saving outputs.
        - device (str): Device to load the model on ('cuda' or 'cpu').
        - sensor_width_mm (float): Sensor width in mm (defaults to 36.0).
        - sensor_height_mm (float): Sensor height in mm (defaults to 24.0).
        """
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm

        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(model_path).to(self.device).eval()
        self.root_folder = Path(root_folder)

    def infer(
            self,
            image: np.ndarray,
            resolution_level: int = 9,
            apply_mask: bool = True,
            fov_x: Optional[float] = None,
            focal_length_mm: Optional[float] = None,
            **model_kwargs
    ) -> MoGeOutput:
        """
        Run inference on a given image array (H, W, 3) in RGB.

        Parameters:
        - image (np.ndarray): The input image as a NumPy array in RGB, shape (H, W, 3).
        - resolution_level (int): Resolution level [0..9] for the inference. Default=9.
        - apply_mask (bool): Whether the model should produce a mask. Default=True.
        - fov_x (float, optional): Known horizontal FOV in degrees. If provided, overrides focal_length_mm.
        - focal_length_mm (float, optional): Known focal length in mm. If provided (and fov_x is not),
          we convert this focal length to fov_x using the sensor_width_mm.
        - **model_kwargs: Any additional keyword arguments supported by the new model.infer().

        Returns:
        - MoGeOutput: An object holding points, depth, mask, intrinsics, etc.
        """
        # If user provided both fov_x & focal_length_mm, we prioritize fov_x
        if fov_x is not None:
            final_fov_x = fov_x
        elif focal_length_mm is not None:
            # Compute fov_x from focal length in mm
            # fov_x (radians) = 2 * arctan( (sensor_width_mm/2) / focal_length_mm )
            # convert to degrees: np.degrees(...)
            final_fov_x = np.degrees(
                2 * np.arctan((self.sensor_width_mm / 2) / focal_length_mm)
            )
        else:
            # Neither passed; let the model estimate
            final_fov_x = None

        # Convert image to torch tensor (C,H,W) in [0..1] range
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        # Model inference
        with torch.no_grad():
            output = self.model.infer(
                image_tensor,
                resolution_level=resolution_level,
                apply_mask=apply_mask,
                fov_x=final_fov_x,  # Pass computed or user-provided value
                **model_kwargs
            )

        # Extract results
        points = output['points'].cpu().numpy()
        depth = output['depth'].cpu().numpy()
        mask = output['mask'].cpu().numpy() if 'mask' in output else None
        intrinsics = output['intrinsics'].cpu().numpy()

        # Create the MoGeOutput object
        inference_output = MoGeOutput(
            points=points,
            depth=depth,
            mask=mask,
            intrinsics=intrinsics,
            image=image,
            root_folder=self.root_folder,
            sensor_width_mm=self.sensor_width_mm,
            sensor_height_mm=self.sensor_height_mm,
            shift=0  # or any shift you need
        )
        return inference_output


if __name__ == '__main__':
    # Example usage
    model_path = r'E:\ai_projects\dust3r_project\MoGo\model\model.pt'
    root_folder = r'E:\ai_projects\dust3r_project\vision-forge\test\output_geo'
    inference = MoGeInference(model_path, root_folder=root_folder)

    # Example image
    img = r'C:/Users/Femto7000/nukesd/SD_Txt2Img/crystalClearXL_ccxl.safetensors/20241116_115144/20241116_115144_1_1.0001.png'
    image_pil = Image.open(img).convert('RGB')
    image_np = np.array(image_pil)

    # Run inference with new argument (e.g. fov_x=50.0).
    # You can also pass other arguments if your model supports them, e.g. threshold=...
    output = inference.infer(
        image_np,
        resolution_level=9,
        apply_mask=False,
        focal_length_mm=75.0
        # any other **model_kwargs
    )

    # Use the output
    camera_data = output.get_camera_data()
    print("Camera Data:", camera_data)

    # Saving various outputs
    output.save_mesh('.output/mesh_filename2', file_format='obj', remove_edge=True, rtol=0.02)
    output.save_camera_nuke('.output/camera')
    output.save_depth_map('.output/depth_map')
    # output.save_frustum_as_obj('frustum/frustum_mesh')

    camera_params = output.get_camera_parameters()
    output.save_camera_nuke('.output/camera2')

    print("Camera Parameters:")
    print(f"Focal Length X (mm): {camera_params['focal_length_x_mm']}")
    print(f"Focal Length Y (mm): {camera_params['focal_length_y_mm']}")
    print(f"Camera Position: {camera_params['position']}")
    print(f"Camera Rotation: {camera_params['rotation']}")
