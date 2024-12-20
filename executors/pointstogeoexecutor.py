import os
import sys

paths = [r'E:\stable-diffusion\stable-diffusion-integrator', r'E:\track_anything_project',
         r'E:\ai_projects\dust3r_project\vision-forge', r'E:\nuke-bridge', 'E:/ai_projects/ai_portal']
for p in paths:
    if not p in sys.path:
        sys.path.append(p)
from nukebridge.executors.core.baseexecutor import BaseExecutor

from geoanything.executors.utils.mesh_generator import MeshGenerator


class PointsToGeoExecutor(BaseExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_parser(self):
        super().setup_parser()
        self.remove_argument('--output')
        self.parser.add_argument('--output_geo', required=True, help='Output path for the generated geometry')
        # self.args['colors'] = self.args['inputs'][list(self.args['inputs'].keys())[0]]
        #

    def run(self):
        self.logger.info(f"Running {self.__class__.__name__} with arguments: {self.dict_to_string(self.args_dict)}")
        input_filename_pattern = self.args_dict['inputs']['positions']
        color_filename_pattern = self.args_dict['inputs']['colors']
        output_filename_pattern = self.args_dict['output_geo']
        frame_range = range(self.frame_range[0], self.frame_range[-1] + 1)

        output_dir = os.path.dirname(output_filename_pattern)
        os.makedirs(output_dir, exist_ok=True)
        points_list = self.imageIO.read_image(input_filename_pattern, frame_range=self.frame_range, output_format='np')
        color_list = self.imageIO.read_image(color_filename_pattern, frame_range=self.frame_range, output_format='np')

        data = self.args_dict.get('data', {})
        output_data = {}
        for positions, colors, frame_number in self.logger.progress(zip(points_list, color_list, frame_range),
                                                                    desc="Processing frames:"):

            mesh_generator = MeshGenerator.from_np(positions, colors=colors)

            # size = points.shape[:2]
            # uvs = self.generate_uvs_from_image(size)

            # r, g, b, a = points[:, :, 0], points[:, :, 1], points[:, :, 2], points[:, :, 3]
            # mask = a > 0  # Filter out points with alpha > 0
            # Create an array of vectors for points with alpha > 0
            # points_np = np.vstack((r[mask], g[mask], b[mask])).T
            # uvs = uvs[mask]
            input_filename = input_filename_pattern % frame_number
            output_filename = output_filename_pattern % frame_number
            output_filename = output_filename if output_filename.endswith('.obj') else output_filename + '.obj'

            if not os.path.exists(input_filename):
                self.logger.warning(f"File not found: {input_filename}")
                continue
            # # Create Open3D PointCloud
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_np)

            # mesh_generator = MeshGenerator(pcd, uvs=uvs)

            # Perform cleaning methods based on data dictionary
            # 1. Clean Point Cloud Methods
            cleaning_methods = data.get('cleaning_methods', {})
            if cleaning_methods:
                mesh_generator.clean_point_cloud(cleaning_methods)

            # 2. Remove Spikes Dynamic
            if data.get('remove_spikes', False):
                iqr_multiplier = data.get('iqr_multiplier', 1.5)
                k = data.get('k', 30)
                mesh_generator.remove_spikes_dynamic(iqr_multiplier=iqr_multiplier, k=k)

            # 3. Remove Edge Noise
            if data.get('remove_edge_noise', False):
                curvature_threshold = data.get('curvature_threshold', 0.1)
                k_curvature = data.get('k_curvature', 30)
                mesh_generator.remove_edge_noise(curvature_threshold=curvature_threshold, k=k_curvature)

            # Generate the mesh
            mesh_params = data.get('mesh_params', {})
            method = mesh_params.get('method', 'poisson')
            if method == 'poisson':
                depth = mesh_params.get('depth', 9)
                width = mesh_params.get('width', 0.0)
                scale = mesh_params.get('scale', 1.0)
                linear_fit = mesh_params.get('linear_fit', False)
                density_threshold = mesh_params.get('density_threshold', 0.0)
                remove_holes = mesh_params.get('remove_holes', False)
                n_threads = mesh_params.get('n_threads', -1)
                regenerate_normals = mesh_params.get('regenerate_normals', False)
                mesh_generator.generate_mesh(method='poisson', depth=depth, width=width, scale=scale,
                                             linear_fit=linear_fit, density_threshold=density_threshold,
                                             remove_holes=remove_holes, n_threads=n_threads,
                                             regenerate_normals=regenerate_normals)
            elif method == 'bpa':
                radius = mesh_params.get('radius', 0.05)
                radii = mesh_params.get('radii', None)
                compute_normals = mesh_params.get('compute_normals', True)
                mesh_generator.generate_mesh(method='bpa', radius=radius, radii=radii,
                                             compute_normals=compute_normals)
            else:
                self.logger.error(f"Unknown meshing method: {method}")
                continue

            # Export the mesh
            mesh_generator.export_obj(output_filename)
            self.logger.info(f'Processed frame {frame_number} saved to {output_filename}')
            output_data[frame_number] = output_filename
        self.send_output(output_data)


if __name__ == '__main__':
    import sys

    skip_parser = len(sys.argv) == 1  # sys.argv includes the script name, so 1 means no additional args
    executor = PointsToGeoExecutor(skip_setup_parser=skip_parser)
    print('starting' * 10)
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
                        'use_statistical': False,
                        'nb_neighbors': 30,
                        'std_ratio': 2.0
                    },
                    'radius': {
                        'use_radius': True,
                        'nb_points': 16,
                        'radius': 5.0
                    }
                },
                # Remove Spikes Dynamic
                'remove_spikes': False,
                'iqr_multiplier': 1.5,
                'k': 30,
                # Remove Edge Noise
                'remove_edge_noise': False,
                'curvature_threshold': 0.1,
                'k_curvature': 30,
                # Mesh Generation Parameters
                'mesh_params': {
                    'method': 'poisson',
                    'depth': 9,
                    'width': 0.0,
                    'scale': 1.0,
                    'linear_fit': True,
                    'density_threshold': 0.0,
                    'remove_holes': False,
                    'n_threads': -1
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
