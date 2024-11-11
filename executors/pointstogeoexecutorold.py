import os
import open3d as o3d

import numpy as np
from nukebridge.executors.core.baseexecutor import BaseExecutor

from geoanything.executors.utils.mesh_generator import MeshGenerator


class PointsToGeoExecutor(BaseExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_parser(self):
        super().setup_parser()
        self.remove_argument('--output')
        self.parser.add_argument('--output_geo', required=True, help='Output path for the generated geometry')

    def run(self):
        self.logger.info(f"Running {self.__class__.__name__} with arguments: {self.dict_to_string(self.args_dict)}")
        input_filename_pattern = self.args_dict['inputs']['positions']
        output_filename_pattern = self.args_dict['output_geo']
        frame_range = range(self.frame_range[0], self.frame_range[-1] + 1)

        output_dir = os.path.dirname(output_filename_pattern)
        os.makedirs(output_dir, exist_ok=True)
        points_list = self.imageIO.read_image(input_filename_pattern, frame_range=self.frame_range, output_format='np')

        data = self.args_dict.get('data', {})

        for points, frame_number in self.logger.progress(zip(points_list, frame_range), desc="Processing frames:"):
            size = points.shape[:2]
            uvs = self.generate_uvs_from_image(size)

            r, g, b, a = points[:, :, 0], points[:, :, 1], points[:, :, 2], points[:, :, 3]
            mask = a > 0  # Filter out points with alpha > 0
            # Create an array of vectors for points with alpha > 0
            points_np = np.vstack((r[mask], g[mask], b[mask])).T
            uvs = uvs[mask]

            # Create Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)

            input_filename = input_filename_pattern % frame_number
            output_filename = output_filename_pattern % frame_number
            output_filename = output_filename if output_filename.endswith('.obj') else output_filename + '.obj'

            if not os.path.exists(input_filename):
                self.logger.warning(f"File not found: {input_filename}")
                continue

            mesh_generator = MeshGenerator(pcd, uvs=uvs)

            # Perform cleaning methods based on data dictionary
            cleaning_methods = data.get('cleaning_methods', {})
            if cleaning_methods:
                mesh_generator.clean_point_cloud(cleaning_methods)

            # Remove spikes dynamic
            if data.get('remove_spikes', False):
                iqr_multiplier = data.get('iqr_multiplier', 1.5)
                k = data.get('k', 30)
                mesh_generator.remove_spikes_dynamic(iqr_multiplier=iqr_multiplier, k=k)

            # Remove edge noise
            if data.get('remove_edge_noise', False):
                curvature_threshold = data.get('curvature_threshold', 0.1)
                k_curvature = data.get('k_curvature', 30)
                mesh_generator.remove_edge_noise(curvature_threshold=curvature_threshold, k=k_curvature)

            # Generate the mesh
            mesh_params = data.get('mesh_params', {})
            depth = mesh_params.get('depth', 9)
            width = mesh_params.get('width', 0)
            scale = mesh_params.get('scale', 1.0)
            linear_fit = mesh_params.get('linear_fit', False)
            mesh_generator.generate_mesh(depth=depth, width=width, scale=scale, linear_fit=linear_fit)

            # Export the mesh
            mesh_generator.export_obj(output_filename)
            self.logger.info(f'Processed frame {frame_number} saved to {output_filename}')


    @staticmethod
    def generate_uvs_from_image(size):
        """
        Generate UV coordinates for each point based on their image coordinates.

        Args:
            size (tuple): The dimensions of the image (height, width).

        Returns:
            np.array: Array of UV coordinates with shape (n, 2).
        """
        # Get the height and width from the size tuple
        height, width = size
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Normalize these coordinates to the range [0, 1]
        u_coords = x_indices.astype(float) / (width - 1)
        v_coords = y_indices.astype(float) / (height - 1)
        v_coords = 1 - v_coords  # Flip the V coordinates to match the image origin
        uvs = np.stack((u_coords, v_coords), axis=-1)

        return uvs


if __name__ == '__main__':
    import sys

    skip_parser = len(sys.argv) == 1  # sys.argv includes the script name, so 1 means no additional args
    executor = PointsToGeoExecutor(skip_setup_parser=skip_parser)

    if skip_parser:
        executor.args_dict = {
            'inputs': {
                'positions': 'C:/Users/Femto7000/geoanything/test2/GAI_PointCloudGen1/20240925/sources/depth_to_position.%04d.exr'
                # Example pattern for input
            },
            'output_geo': 'C:/Users/Femto7000/geoanything/test2/GAI_PointCloudGen1/20240925/20240925_072242/output_geo.%04d.obj',
            # Example pattern for output files
            'frame_range': (1001, 1001),
            'data': {
                # Clean Point Cloud Methods
                'cleaning_methods': {
                    'statistical': {
                        'nb_neighbors': 20,
                        'std_ratio': 2.0
                    },
                    'radius': {
                        'nb_points': 16,
                        'radius': 0.05
                    }
                },
                # Remove Spikes Dynamic
                'remove_spikes': True,
                'iqr_multiplier': 1.5,
                'k': 30,
                # Remove Edge Noise
                'remove_edge_noise': True,
                'curvature_threshold': 0.1,
                'k_curvature': 30,
                # Mesh Generation Parameters
                'mesh_params': {
                    'depth': 9,
                    'width': 0,
                    'scale': 1.0,
                    'linear_fit': False
                }
            }
        }

    try:
        # Try to get the logger level from the arguments.
        lvl = int(executor.args_dict.get('logger_level', 10))
    except TypeError:
        # If the logger level is not specified in the arguments, set it to 10.
        lvl = 10
    # Set the logger level.
    executor.logger.setLevel(lvl)
    executor.run()
