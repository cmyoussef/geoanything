# mesh_generator.py

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import iqr


class MeshGenerator:

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

    @classmethod
    def from_np(cls, positions, uvs=None, colors=None):
        """
        Initialize the MeshGenerator with a NumPy array of points, optional UV coordinates, and optional vertex colors.

        Parameters:
            positions (np.array): The points to process (N x 3).
            uvs (np.array): The UV coordinates for each point (N x 2).
            colors (np.array): The RGB colors for each point (N x 3), values in range [0, 1].
        """

        r, g, b, a = positions[:, :, 0], positions[:, :, 1], positions[:, :, 2], positions[:, :, 3]
        mask = a > 0  # Filter out points with alpha > 0

        if uvs is None:
            print("UVs not provided. Generating UVs from image size.", positions.shape[:2])
            uvs = cls.generate_uvs_from_image(positions.shape[:2])
            uvs = uvs[mask]

        points = np.vstack((r[mask], g[mask], b[mask])).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            r_c, g_c, b_c = colors[:, :, 0], colors[:, :, 1], colors[:, :, 2]
            colors = np.vstack((r_c[mask], g_c[mask], b_c[mask])).T
            colors = colors.astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return cls(pcd, uvs)

    def __init__(self, pcd: o3d.geometry.PointCloud, uvs=None):
        """
        Initialize the MeshGenerator with an Open3D PointCloud, optional UV coordinates, and optional vertex colors.

        Parameters:
            pcd (o3d.geometry.PointCloud): The point cloud to process.
            uvs (np.array): The UV coordinates for each point (N x 2).
            colors (np.array): The RGB colors for each point (N x 3), values in range [0, 1].
        """
        self.original_pcd = pcd
        self.pcd = o3d.geometry.PointCloud(pcd)
        self.uvs = uvs
        self.original_uvs = uvs.copy() if uvs is not None else None
        self.mesh = None
        self.mesh_uvs = None
        self.mesh_colors = None  # Will be set later
        self.scale = self.get_scale()
        self.num_points = len(self.pcd.points)

    def get_scale(self):
        bbox = self.pcd.get_axis_aligned_bounding_box()
        scale = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        return scale

    def reset_pcd(self):
        """
        Reset the point cloud to the original point cloud.
        """
        self.pcd = o3d.geometry.PointCloud(self.original_pcd)
        # if self.uvs is not None:
        #     self.uvs = self.original_uvs.copy()

    def assign_normals_to_mesh(self):
        """
        Assign normals to the mesh vertices by interpolating from the point cloud normals.
        """
        if self.mesh is None:
            print("Mesh not generated yet. Cannot assign normals to mesh.")
            return

        # Ensure the point cloud has normals
        if not self.pcd.has_normals():
            self.pcd.estimate_normals()
            self.pcd.orient_normals_consistent_tangent_plane(10)

        # Get the mesh vertices
        mesh_vertices = np.asarray(self.mesh.vertices)

        # Build a KDTree from the point cloud
        pcd_points = np.asarray(self.pcd.points)
        pcd_normals = np.asarray(self.pcd.normals)
        tree = cKDTree(pcd_points)

        # For each mesh vertex, find the nearest point in the point cloud
        _, indices = tree.query(mesh_vertices, k=1)

        # Assign normals to the mesh vertices
        self.mesh.vertex_normals = o3d.utility.Vector3dVector(pcd_normals[indices])

    def clean_point_cloud(self, method_params):
        """
        Clean the point cloud using Open3D's built-in methods based on the provided parameters.

        Parameters:
            method_params (dict): Dictionary containing method names as keys and their parameters as values.
        """
        for method_name, params in method_params.items():
            if method_name == 'statistical' and params.get('use_statistical', False):
                nb_neighbors = params.get('nb_neighbors', .01)
                if nb_neighbors < 1:
                    nb_neighbors = max(3, int(self.num_points * nb_neighbors))\

                std_ratio = params.get('std_ratio', 2.0)
                _, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio,
                                                             print_progress=True)
                self.pcd = self.pcd.select_by_index(ind)
                # if self.uvs is not None:
                #     self.uvs = self.original_uvs[ind]
            elif method_name == 'radius' and params.get('use_radius', False):
                nb_points = params.get('nb_points', .01)
                if nb_points < 1:
                    nb_points = max(5, int(self.num_points * nb_points))

                radius = params.get('radius', 0.05)
                if radius > self.scale:
                    print(f"Radius:{radius} is greater than the scale:{self.scale} of the point cloud. Using 5% of the scale as radius.")
                    radius = self.scale * 0.05

                _, ind = self.pcd.remove_radius_outlier(nb_points=nb_points, radius=radius, print_progress=True)
                self.pcd = self.pcd.select_by_index(ind)
                # if self.uvs is not None:
                #     self.uvs = self.original_uvs[ind]
            elif method_name not in ['statistical', 'radius']:
                print(f"Unknown cleaning method: {method_name}")

    def remove_spikes_dynamic(self, iqr_multiplier=1.5, k=30):
        """
        Dynamically remove spikes from the point cloud based on the statistical properties of nearest neighbor distances.
        @param iqr_multiplier: Multiplier for the IQR to set threshold for spike removal.
        @param k: Number of nearest neighbors to consider for each point.
        """
        points = np.asarray(self.pcd.points)
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k + 1)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        dist_iqr = iqr(mean_distances)
        threshold = np.median(mean_distances) + dist_iqr * iqr_multiplier
        valid_indices = mean_distances < threshold
        self.pcd = self.pcd.select_by_index(np.where(valid_indices)[0])
        # if self.uvs is not None:
        #     self.uvs = self.original_uvs[valid_indices]

    def remove_edge_noise(self, curvature_threshold=0.1, k=30):
        """
        Remove points around edges by analyzing curvature.

        Parameters:
            curvature_threshold (float): Threshold for curvature above which points are considered noise.
            k (int): Number of nearest neighbors to use for normal and curvature estimation.
        """
        # Estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

        # Compute curvature
        curvatures = self.compute_curvature(self.pcd, k)

        # Identify points with curvature below the threshold
        valid_indices = curvatures < curvature_threshold

        # Update point cloud and UVs
        self.pcd = self.pcd.select_by_index(np.where(valid_indices)[0])
        # if self.original_uvs is not None:
        #     self.uvs = self.original_uvs[valid_indices]

    def assign_colors_to_mesh(self):
        """
        Assign colors to the mesh vertices by interpolating from the point cloud colors.
        """
        if self.mesh is None:
            print("Mesh not generated yet. Cannot assign colors to mesh.")
            return

        if not self.pcd.has_colors():
            print("Point cloud does not have colors. Cannot assign colors to mesh.")
            return

        # Get the mesh vertices
        mesh_vertices = np.asarray(self.mesh.vertices)

        # Build a KDTree from the point cloud
        pcd_points = np.asarray(self.pcd.points)
        pcd_colors = np.asarray(self.pcd.colors)
        tree = cKDTree(pcd_points)

        # For each mesh vertex, find the nearest point in the point cloud
        _, indices = tree.query(mesh_vertices, k=1)

        # Assign colors to the mesh vertices
        self.mesh_colors = pcd_colors[indices]

    @staticmethod
    def compute_curvature(pcd, k):
        """
        Compute curvature for each point in the point cloud.

        Parameters:
            pcd (o3d.geometry.PointCloud): The point cloud with normals estimated.
            k (int): Number of nearest neighbors to use for curvature estimation.

        Returns:
            np.array: Array of curvature values for each point.
        """
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        tree = o3d.geometry.KDTreeFlann(pcd)

        curvatures = np.zeros(len(points))
        for i in range(len(points)):
            _, idx, _ = tree.search_knn_vector_3d(points[i], k)
            neighbor_normals = normals[idx[1:]]  # Exclude the point itself
            normal_diffs = neighbor_normals - normals[i]
            curvature = np.linalg.norm(normal_diffs, axis=1).mean()
            curvatures[i] = curvature

        return curvatures

    def generate_mesh(self, method='poisson', **kwargs):
        if method == 'poisson':
            self.generate_mesh_poisson(**kwargs)
        elif method == 'bpa':
            self.generate_mesh_bpa(**kwargs)
        else:
            print(f"Unknown meshing method: {method}")

    def generate_mesh_bpa(self, radius=0.05, radii=None, compute_normals=True):
        if compute_normals and not self.pcd.has_normals():
            self.pcd.estimate_normals()

        if radii is None:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                self.pcd, o3d.utility.DoubleVector([radius])
            )
        else:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                self.pcd, o3d.utility.DoubleVector(radii)
            )

        self.mesh = mesh

        # Assign UVs, normals, and colors to the mesh vertices
        self.assign_uvs_to_mesh()
        self.assign_normals_to_mesh()
        self.assign_colors_to_mesh()

    def generate_mesh_poisson(self, depth=8, width=0, scale=1.1, linear_fit=False,
                              n_threads=-1, density_threshold=0.0, remove_holes=False, regenerate_normals=False):
        """
        Generate a mesh using Poisson surface reconstruction.

        Parameters:
            depth (int): Maximum depth of the tree (controls mesh resolution).
            width (float): Target width of the finest level octree cells.
            scale (float): Ratio between reconstruction cube and samples' bounding cube.
            linear_fit (bool): Use linear interpolation for iso-vertex positions.
            n_threads (int): Number of threads to use (-1 uses all available cores).
            density_threshold (float): Threshold to remove low-density vertices.
            remove_holes (bool): Fill small holes in the mesh.
            regenerate_normals (bool): Regenerate normals before meshing.
        """
        # Ensure normals are estimated
        # Assign normals to the mesh vertices
        if regenerate_normals or not self.pcd.has_normals():
            self.pcd.estimate_normals()
            self.pcd.orient_normals_consistent_tangent_plane(10)

        # Perform Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit, n_threads=n_threads
        )

        # Optionally remove low-density vertices
        # Optionally remove low-density vertices
        if density_threshold > 0.0:
            densities = np.asarray(densities)
            density_threshold_value = np.quantile(densities, density_threshold)
            vertices_to_remove = densities < density_threshold_value
            mesh.remove_vertices_by_mask(vertices_to_remove)
        else:
            # Default behavior: remove the bottom 1% by density
            densities = np.asarray(densities)
            density_threshold_value = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold_value
            mesh.remove_vertices_by_mask(vertices_to_remove)

        # Optionally remove small isolated pieces and fill holes
        if remove_holes:
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

        self.mesh = mesh
        self.mesh_densities = densities

        # Assign UVs, normals, and colors to the mesh vertices
        self.assign_uvs_to_mesh()
        self.assign_normals_to_mesh()
        self.assign_colors_to_mesh()

    def assign_uvs_to_mesh(self):
        """
        Assign UV coordinates to the mesh vertices by interpolating from the point cloud UVs.
        """
        if self.mesh is None:
            print("Mesh not generated yet. Cannot assign UVs to mesh.")
            return

        if self.uvs is None:
            print("UVs not provided. Cannot assign UVs to mesh.")
            return

        # Get the mesh vertices
        mesh_vertices = np.asarray(self.mesh.vertices)

        # Build a KDTree from the point cloud
        # pcd_points = np.asarray(self.pcd.points)
        tree = cKDTree(self.original_pcd.points)

        # For each mesh vertex, find the nearest point in the point cloud
        _, indices = tree.query(mesh_vertices, k=1)

        # Assign UVs to the mesh vertices
        self.mesh_uvs = self.original_uvs[indices]

    def export_obj(self, filepath):
        """
        Export the mesh as an OBJ file with vertices, UVs, normals, colors, and faces.

        Parameters:
            filepath (str): The path to save the OBJ file.
        """
        if self.mesh is None:
            print("Mesh not generated yet. Please call generate_mesh() first.")
            return

        # Get the mesh data
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)
        uvs = self.mesh_uvs if self.mesh_uvs is not None else np.zeros((len(vertices), 2))
        normals = np.asarray(self.mesh.vertex_normals)
        colors = self.mesh_colors if self.mesh_colors is not None else None

        with open(filepath, 'w') as file:
            # Write normals
            for n in normals:
                file.write(f"vn {n[0]} {n[1]} {n[2]}\n")
            # Write vertices (with colors if available)
            if colors is not None:
                for v, c in zip(vertices, colors):
                    file.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
            else:
                for v in vertices:
                    file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # Write UVs
            for uv in uvs:
                file.write(f"vt {uv[0]} {uv[1]}\n")
            # Write faces
            for f in faces:
                idx = f + 1  # OBJ format is 1-indexed
                file.write(f"f {idx[0]}/{idx[0]}/{idx[0]} {idx[1]}/{idx[1]}/{idx[1]} {idx[2]}/{idx[2]}/{idx[2]}\n")
