import sys
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
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from PIL import Image
import trimesh
import trimesh.visual

# MoGe / 3D utilities
from MoGe.moge.model import MoGeModel
from MoGe.moge.utils.vis import colorize_depth
from MoGe import utils3d

# For multi-layer EXR writing
from nukebridge.utils.image_io import get_image_io

# Official panorama pipeline helpers
from MoGe.scripts.infer_panorama import (
    get_panorama_cameras,
    split_panorama_image,
    merge_panorama_depth,
    spherical_uv_to_directions
)

def srgb_to_linear(img_srgb_8u: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 sRGB image [0..255] to float32 linear [0..1].
    Uses the standard sRGB -> linear piecewise transform.
    Applies channel-wise.
    """
    # Ensure float in [0..1]
    img_float = img_srgb_8u.astype(np.float32) / 255.0

    # piecewise transform
    # v_lin = v_srgb / 12.92 if v_srgb <= 0.04045
    # v_lin = ((v_srgb+0.055)/1.055)^2.4 otherwise
    mask_low = img_float <= 0.04045
    img_linear = np.empty_like(img_float, dtype=np.float32)

    img_linear[mask_low] = (img_float[mask_low] / 12.92)
    img_linear[~mask_low] = ((img_float[~mask_low] + 0.055) / 1.055) ** 2.4
    return img_linear

class MoGePanoramaInferenceRefactored:
    """
    A class to run panorama inference on a single equirectangular image, then:
      - split -> MoGe infer subviews -> Poisson-merge
      - optionally rotate geometry and flip UV
      - produce a multi-layer EXR with depth/mask/normals/points/rgba/depth_vis
      - produce a single OBJ mesh with texture
    """

    def __init__(
        self,
        model_path: str,
        root_folder: str = '.',
        device: str = 'cuda'
    ):
        """
        :param model_path: Local path or HF name of the MoGe model.
        :param root_folder: Base folder where outputs (.exr, .obj) are written.
        :param device: 'cuda', 'cuda:0', or 'cpu'
        """
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(model_path).to(self.device).eval()
        self.root_folder = Path(root_folder)
        # For writing multi-layer EXR
        self.image_io = get_image_io()

    def infer_panorama(
        self,
        input_image_path: str,
        output_exr_name: str = 'layers.exr',
        output_obj_name: str = 'mesh.obj',
        resize_to: Optional[int] = None,
        resolution_level: int = 9,
        threshold: float = 0.03,
        batch_size: int = 4,
        remove_edges: bool = True,
        rotate_x_neg90: bool = False,
        flip_v: bool = False
    ):
        """
        Runs the panorama pipeline on a single input image path.

        Args:
            input_image_path: Path to a single equirectangular image (jpg/png/etc.).
            output_exr_name: File name for the multi-layer EXR (within root_folder).
            output_obj_name: File name for the OBJ mesh (within root_folder).
            resize_to: If provided, resizes the input so that max dimension <= this value.
            resolution_level: MoGe inference resolution [0..9].
            threshold: Threshold for removing edges (depth/normal) from final mesh.
            batch_size: How many splitted sub-views to process in each MoGe forward pass.
            remove_edges: If True, apply edge removal to the final mesh.
            rotate_x_neg90: If True, rotate final geometry -90° about X axis (and also the 3D points).
            flip_v: If True, flip the V coordinate in the UV.
        """

        # 1) Load input
        image_path = Path(input_image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        image_rgb_8u = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image_rgb_8u.shape[:2]

        # 2) Optional resize
        if resize_to is not None:
            scale = min(resize_to / float(orig_height), resize_to / float(orig_width))
            new_h, new_w = int(orig_height * scale), int(orig_width * scale)
            image_rgb_8u = cv2.resize(image_rgb_8u, (new_w, new_h), interpolation=cv2.INTER_AREA)
        height, width = image_rgb_8u.shape[:2]

        # 3) Split
        splitted_extrinsics, splitted_intrinsics = get_panorama_cameras()
        splitted_resolution = 512  # from reference
        splitted_images = split_panorama_image(
            image_rgb_8u,
            splitted_extrinsics,
            splitted_intrinsics,
            splitted_resolution
        )

        # 4) MoGe inference on splitted sub-views
        splitted_distance_maps = []
        splitted_masks = []
        for start_idx in range(0, len(splitted_images), batch_size):
            batch = splitted_images[start_idx : start_idx + batch_size]
            batch_np = np.stack(batch).astype(np.float32) / 255.0
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=self.device).permute(0,3,1,2)

            with torch.no_grad():
                out = self.model.infer(
                    batch_tensor,
                    resolution_level=resolution_level,
                    apply_mask=False
                )
                pts = out['points'].cpu()  # shape (B,H_sub,W_sub,3)
                dist_maps = pts.norm(dim=-1).numpy()   # shape (B,H_sub,W_sub)
                mask_maps = out['mask'].cpu().numpy()  # shape (B,H_sub,W_sub)
            splitted_distance_maps.extend(list(dist_maps))
            splitted_masks.extend(list(mask_maps))

        # 5) Poisson merge
        merging_width, merging_height = min(1920, width), min(960, height)
        panorama_depth, panorama_mask = merge_panorama_depth(
            merging_width,
            merging_height,
            splitted_distance_maps,
            splitted_masks,
            splitted_extrinsics,
            splitted_intrinsics
        )
        # Upsample to final size
        panorama_depth = panorama_depth.astype(np.float32)
        panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
        panorama_mask = cv2.resize(
            panorama_mask.astype(np.uint8),
            (width, height),
            cv2.INTER_NEAREST
        ).astype(bool)

        # 6) Build final 3D points
        uv = utils3d.numpy.image_uv(width=width, height=height)
        directions = spherical_uv_to_directions(uv)
        points_3d = panorama_depth[...,None] * directions  # (H,W,3)

        # 7) Compute normals for edge removal
        normals, normals_mask = utils3d.numpy.points_to_normals(points_3d, panorama_mask)

        # 8) If user wants edge removal
        if remove_edges:
            remove_edges_mask = ~(
                utils3d.numpy.depth_edge(panorama_depth, rtol=threshold)
                & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)
            )
            final_mask = panorama_mask & remove_edges_mask
        else:
            final_mask = panorama_mask

        # 9) Mesh creation
        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points_3d,
            (image_rgb_8u.astype(np.float32) / 255.0),
            uv,
            mask=final_mask,
            tri=True
        )

        # 10) Rotation if requested
        # For -90° around X: (x,y,z) -> (x, -z, y)
        Rx_neg90 = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ], dtype=np.float32)

        if rotate_x_neg90:
            # Rotate vertices
            vertices = vertices @ Rx_neg90.T
            # Also rotate the entire 3D point cloud
            points_reshaped = points_3d.reshape(-1,3) @ Rx_neg90.T
            points_3d = points_reshaped.reshape(height, width, 3)
            # Also rotate the normals if you want them consistent
            # normals_reshaped = normals.reshape(-1,3) @ Rx_neg90.T
            # normals = normals_reshaped.reshape(height, width, 3)

        # 11) Flip V if requested
        if flip_v:
            vertex_uvs[...,1] = 1.0 - vertex_uvs[...,1]

        # 12) Prepare output paths
        self.root_folder.mkdir(parents=True, exist_ok=True)
        exr_path = self.root_folder / output_exr_name
        obj_path = self.root_folder / output_obj_name

        # 13) Build a multi-layer EXR
        # We'll store:
        #   depth, normals, points, plus an RGBA (where A = mask),
        #   plus a depth_vis in linear space,
        #   all in float32.

        # We do sRGB -> linear for the input image
        # Then store mask as alpha
        rgba_linear = np.zeros((height, width, 4), dtype=np.float32)
        rgba_linear[...,:3] = srgb_to_linear(image_rgb_8u)
        rgba_linear[...,3] = panorama_mask.astype(np.float32)  # alpha=mask

        # Also build depth_vis as sRGB -> linear
        depth_vis_8u = colorize_depth(panorama_depth, panorama_mask)  # shape(H,W,3) in 8-bit sRGB
        depth_vis_linear = srgb_to_linear(depth_vis_8u)

        # channels:
        #   rgba.R/G/B/A
        #   depth.R
        #   normals.(R,G,B)
        #   points.(R,G,B)
        #   depth_vis.(R,G,B)
        layers_dict = {
            "rgba.R":      rgba_linear[...,0],
            "rgba.G":      rgba_linear[...,1],
            "rgba.B":      rgba_linear[...,2],
            "rgba.A":      rgba_linear[...,3],

            "depth.R":     panorama_depth,

            "normals.R":   normals[...,0],
            "normals.G":   normals[...,1],
            "normals.B":   normals[...,2],

            "points.R":    points_3d[...,0],
            "points.G":    points_3d[...,1],
            "points.B":    points_3d[...,2],

            "depth_vis.R": depth_vis_linear[...,0],
            "depth_vis.G": depth_vis_linear[...,1],
            "depth_vis.B": depth_vis_linear[...,2],
        }

        self.image_io.write_multilayer(layers_dict, str(exr_path), image_format='exr')
        print(f"Multi-layer EXR saved to: {exr_path}")

        # 14) Save OBJ
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=trimesh.visual.TextureVisuals(
                uv=vertex_uvs,
                image=Image.fromarray(image_rgb_8u)
            ),
            process=False
        )
        mesh.export(str(obj_path))
        print(f"OBJ mesh saved to: {obj_path}")

        print(f"Done. Wrote EXR -> {exr_path}, OBJ -> {obj_path}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    model_path = r'E:\ai_projects\dust3r_project\MoGo\model\model.pt'
    panorama_infer = MoGePanoramaInferenceRefactored(
        model_path=model_path,
        root_folder=r'E:\ai_projects\dust3r_project\vision-forge\test\output_panorama',
        device='cuda'
    )

    # Single image usage
    input_img = r'E:\ai_projects\dust3r_project\MoGo\MoGe\example_images\Braunschweig_Panoram.jpg'
    panorama_infer.infer_panorama(
        input_image_path=input_img,
        output_exr_name='my_layers.exr',
        output_obj_name='my_mesh.obj',
        resize_to=None,
        resolution_level=9,
        threshold=0.03,
        batch_size=4,
        remove_edges=True,
        rotate_x_neg90=True,
        flip_v=True
    )
