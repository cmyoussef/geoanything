import os

import numpy as np
from nukebridge.executors.core.baseexecutor import BaseExecutor, dict_to_string
from scipy.spatial import cKDTree, Delaunay
from scipy.stats import iqr
from sklearn.cluster import DBSCAN


class PointCloudProcessor:
    def __init__(self, points: np.array, eps: float = 0.5, min_samples: int = 10, k: int = 20,
                 iqr_multiplier: float = 1.5, uvs=None):
        """
        Initialize the PointCloudProcessor with a set of points and cleaning parameters.

        Parameters:
            points (np.array): xyz The array of points to process.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Used in DBSCAN.
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point. Used in DBSCAN.
            k (int): The number of nearest neighbors to consider for dynamic spike removal.
            iqr_multiplier (float): Multiplier for the interquartile range to set dynamic spike removal threshold.
        """

        self.points = points
        self.clean_points = self.points.copy()
        self.min_x, self.min_y = np.min(self.clean_points, axis=0)[:2]
        self.max_x, self.max_y = np.max(self.clean_points, axis=0)[:2]
        self.uvs = uvs if uvs is not None else self.generate_uvs_from_points()

        self.eps = eps
        self.min_samples = min_samples
        self.k = k
        self.iqr_multiplier = iqr_multiplier

    def generate_uvs_from_points(self, recalculate_range=False):
        """
        Normalize the X and Y coordinates of clean points to [0, 1] to generate UV coordinates.
        @param recalculate_range: If True, recalculate the min and max values for normalization.
        """
        if recalculate_range:
            min_x, min_y = np.min(self.clean_points, axis=0)[:2]
            max_x, max_y = np.max(self.clean_points, axis=0)[:2]
        else:
            min_x, min_y = self.min_x, self.min_y
            max_x, max_y = self.max_x, self.max_y

        normalized = np.zeros((self.clean_points.shape[0], 2))
        normalized[:, 0] = (self.clean_points[:, 0] - min_x) / (max_x - min_x)
        normalized[:, 1] = (self.clean_points[:, 1] - min_y) / (max_y - min_y)
        return normalized

    def export_obj(self, filepath):
        """
        Export the cleaned points as an OBJ file with vertices, UVs, and faces.

        Parameters:
            filepath (str): The path to save the OBJ file.
        """
        self.clean_points = self.clean_points.reshape(-1, 3)  # Ensuring it is reshaped to (n, 3) if n points exist
        clean_points_2d = np.delete(self.clean_points, 1, axis=1)
        tri = Delaunay(clean_points_2d)
        faces = tri.simplices
        with open(filepath, 'w') as file:
            for uv in self.uvs:
                file.write(f"vt {uv[0]} {uv[1]}\n")
            for v in self.clean_points:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in faces:
                file.write(f"f {f[0] + 1}/{f[0] + 1} {f[1] + 1}/{f[1] + 1} {f[2] + 1}/{f[2] + 1}\n")

    def remove_spikes_dynamic(self):
        """
        Dynamically remove spikes from the point cloud based on the statistical properties of nearest neighbor distances.
        """
        tree = cKDTree(self.clean_points)
        distances, _ = tree.query(self.clean_points, k=self.k + 1)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        dist_iqr = iqr(mean_distances)
        threshold = np.median(mean_distances) + dist_iqr * self.iqr_multiplier
        valid_indices = mean_distances < threshold
        self.clean_points = self.clean_points[valid_indices]
        self.uvs = self.uvs[valid_indices]

    def enhanced_cleaning(self):
        """
        Apply enhanced cleaning strategies using DBSCAN to remove isolated or stretched points.
        """
        points_2d = np.delete(self.clean_points, 1, axis=1)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_2d)
        labels = clustering.labels_
        valid_indices = labels != -1
        self.clean_points = self.clean_points[valid_indices]
        self.uvs = self.uvs[valid_indices]

    @staticmethod
    def generate_uvs_from_image(size):
        """
        Generate UV coordinates for each point based on their image coordinates.

        Args:
            points (np.array): Array of image points with shape (n, 3) where each row is [r, g, b].
            size (tuple): The dimensions of the image (height, width).

        Returns:
            np.array: Array of UV coordinates with shape (n, 2).
        """
        # Get the height and width from the size tuple
        height, width = size
        print('size', height, width)
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Normalize these coordinates to the range [0, 1]
        u_coords = x_indices.astype(float) / (width - 1)
        v_coords = y_indices.astype(float) / (height - 1)
        v_coords = 1 - v_coords  # Flip the V coordinates to match the image origin
        uvs = np.stack((u_coords, v_coords), axis=-1)

        return uvs


class PointsToGeoExecutor(BaseExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_parser(self):
        super().setup_parser()
        self.parser.add_argument('--enhanced-cleaning', type=bool, default=False,
                                 help='Apply enhanced cleaning using DBSCAN.')
        self.parser.add_argument('--remove-spikes', type=bool, default=False,
                                 help='Remove dynamic spikes from the point cloud.')
        self.parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps parameter.')
        self.parser.add_argument('--min-samples', type=int, default=10, help='DBSCAN min_samples parameter.')
        self.parser.add_argument('--k', type=int, default=20, help='Number of nearest neighbors for spike removal.')
        self.parser.add_argument('--iqr-multiplier', type=float, default=1.5,
                                 help='Multiplier for the IQR to set threshold for spike removal.')

    def run(self):
        self.logger.info(f"Running {self.__class__.__name__} with arguments: {dict_to_string(self.args_dict)}")
        input_filename_pattern = self.args_dict['input']
        output_filename_pattern = self.args_dict['output']
        frame_range = range(self.frame_range[0], self.frame_range[-1] + 1)

        output_dir = os.path.dirname(output_filename_pattern)
        os.makedirs(output_dir, exist_ok=True)
        points_list = self.imageIO.read_image(input_filename_pattern, frame_range=self.frame_range, output_format='np')

        for points, frame_number in self.logger.progress(zip(points_list, frame_range), desc="Processing frames:"):
            size = points.shape[:2]
            uvs = PointCloudProcessor.generate_uvs_from_image(size)

            r, g, b, a = points[:, :, 0], points[:, :, 1], points[:, :, 2], points[:, :, 3]
            mask = a > 0  # Filter out points with alpha 0
            print('mask:', mask.shape, mask.sum())
            print('uvs:', uvs.shape)
            print('a:', a.shape)
            # Create an array of vectors for points with alpha > 0
            points = np.vstack((r[mask], g[mask], b[mask])).T
            uvs = uvs[mask]
            input_filename = input_filename_pattern % frame_number
            output_filename = output_filename_pattern % frame_number
            output_filename = output_filename if output_filename.endswith('.obj') else output_filename + '.obj'

            if not os.path.exists(input_filename):
                self.logger.warning(f"File not found: {input_filename}")
                continue

            points_processor = PointCloudProcessor(points, self.args_dict['eps'], self.args_dict['min-samples'],
                                                   self.args_dict['k'], self.args_dict['iqr-multiplier'], uvs=uvs)

            if self.args_dict['remove-spikes']:
                self.logger.info('Removing spikes.')
                points_processor.remove_spikes_dynamic()

            if self.args_dict['enhanced-cleaning']:
                self.logger.info('Applying enhanced cleaning.')
                points_processor.enhanced_cleaning()


            self.logger.info(f'Cleaned points: {points_processor.clean_points.shape[0]}')
            points_processor.export_obj(output_filename)
            self.logger.info(f'Processed frame {frame_number} saved to {output_filename}')


if __name__ == '__main__':
    import sys

    skip_parser = len(sys.argv) == 1  # sys.argv includes the script name, so 1 means no additional args
    executor = PointsToGeoExecutor(skip_setup_parser=skip_parser)
    # for testing
    executor.args_dict = {
        'input': 'C:/Users/Femto7000/geoanything/porjection/GAI_PointCloudGen/20240704/20240704_065721/output.%04d.exr',
        # Example pattern for input files
        'output': 'C:/Users/Femto7000/geoanything/porjection/GAI_PointCloudGen/20240704/20240704_065721/output.%04d',
        # Example pattern for output files
        'enhanced-cleaning': True,
        'remove-spikes': True,
        'eps': 0.1,
        'min-samples': 2,
        'k': 20,
        'iqr-multiplier': 1.5,
        'frame_range': (1, 1)

    }
    try:
        # Try to get the logger level from the arguments.
        lvl = int(executor.args_dict.get('logger_level', 10))
    except TypeError:
        # If the logger level is not specified in the arguments, set it to 20.
        lvl = 10
    # Set the logger level.
    executor.logger.setLevel(lvl)
    executor.run()
