import sys

import numpy as np
import open3d as o3d
from PySide6 import QtWidgets, QtCore


class Open3DWidget(QtWidgets.QWidget):
    def __init__(self, point_cloud, parent=None):
        super(Open3DWidget, self).__init__(parent)
        self.point_cloud = point_cloud
        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(visible=True)
        render_option = self.vis.get_render_option()
        render_option.point_size = 5.0  # Increase point size for better visibility
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background for better contrast
        render_option.line_width = 2.0  # If you are rendering lines
        render_option.show_coordinate_frame = True  # Show a coordinate frame

        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)  # Zoom in
        view_control.set_front([0.0, 0.0, -1.0])  # Set camera direction
        view_control.set_lookat([0.0, 0.0, 0.0])  # Set lookat point
        view_control.set_up([0.0, -1.0, 0.0])  # Set up direction

        self.img = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(100)  # Update every 100 ms

        self.vis.add_geometry(self.point_cloud)

    def update_visualization(self):
        if self.point_cloud is not None:
            # print("Updating visualization", self.point_cloud)
            self.vis.update_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def set_point_cloud(self, point_cloud):
        """Update the point cloud data without clearing the geometry."""
        if self.point_cloud is not None:
            # Update the points of the existing point cloud
            self.point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points))
            self.vis.update_geometry(self.point_cloud)
        else:
            # Add geometry for the first time if no geometry exists yet
            self.point_cloud = point_cloud
            self.vis.add_geometry(self.point_cloud)

    def closeEvent(self, event):
        self.vis.destroy_window()
        event.accept()


class PointCloudProcessingThread(QtCore.QThread):
    update_point_cloud = QtCore.Signal(o3d.geometry.PointCloud)

    def __init__(self, pcd, uvs, params):
        super().__init__()
        self.pcd = pcd
        self.uvs = uvs
        self.params = params
        self._is_running = True

    def run(self):
        if self._is_running:
            # Create a MeshGenerator instance with current parameters
            mesh_generator = MeshGenerator(self.pcd, uvs=self.uvs)

            # Perform cleaning methods based on params
            # Clean Point Cloud Methods
            cleaning_methods = self.params.get('cleaning_methods', {})
            if cleaning_methods:
                mesh_generator.clean_point_cloud(cleaning_methods)

            # Remove Spikes Dynamic
            if self.params.get('remove_spikes', False):
                iqr_multiplier = self.params.get('iqr_multiplier', 1.5)
                k = self.params.get('k', 30)
                mesh_generator.remove_spikes_dynamic(iqr_multiplier=iqr_multiplier, k=k)

            # Remove Edge Noise
            if self.params.get('remove_edge_noise', False):
                curvature_threshold = self.params.get('curvature_threshold', 0.1)
                k_curvature = self.params.get('k_curvature', 30)
                mesh_generator.remove_edge_noise(curvature_threshold=curvature_threshold, k=k_curvature)

            # Emit the updated point cloud to the main thread
            self.update_point_cloud.emit(mesh_generator.pcd)
            self._is_running = False  # Stop after processing

    def stop(self):
        self._is_running = False
        self.quit()
        self.wait()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_pcd, initial_uvs):
        super().__init__()
        self.setWindowTitle("Point Cloud Visualizer with Open3D")
        self.resize(1200, 800)

        # Store initial data
        self.pcd = initial_pcd
        self.uvs = initial_uvs
        voxel_size = 0.05  # Adjust as needed
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        # Set up UI
        self.setup_ui()

        # Start the initial visualization
        # self.update_point_cloud(self.pcd)
        # self.update_point_cloud(initial_pcd)
        # Default parameters
        self.params = {
            'cleaning_methods': {
                'statistical': {
                    'use_statistical': False,
                    'nb_neighbors': 30,
                    'std_ratio': 2.0
                },
                'radius': {
                    'use_radius': False,
                    'nb_points': 16,
                    'radius': 5
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
                'depth': 9,
                'width': 0,
                'scale': 1.0,
                'linear_fit': True
            }
        }

        # Set up UI
        self.setup_ui()
        # self.update_point_cloud(self.pcd)
        # Start the initial visualization
        # self.update_point_cloud(self.points)

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left side: Parameters
        params_layout = QtWidgets.QVBoxLayout()
        self.setup_parameters_ui(params_layout)
        main_layout.addLayout(params_layout)

        # Right side: Open3D Visualization
        self.visualizer = Open3DWidget(self.pcd, parent=self)
        # main_layout.addWidget(self.visualizer)

    def setup_parameters_ui(self, layout):
        # Statistical Cleaning Parameters
        stat_group = QtWidgets.QGroupBox("Statistical Cleaning")
        stat_layout = QtWidgets.QFormLayout()

        self.use_statistical_cb = QtWidgets.QCheckBox("Use Statistical Cleaning")
        self.use_statistical_cb.stateChanged.connect(self.on_parameters_changed)
        stat_layout.addRow(self.use_statistical_cb)

        self.nb_neighbors_spin = QtWidgets.QSpinBox()
        self.nb_neighbors_spin.setRange(1, 100)
        self.nb_neighbors_spin.setValue(30)
        self.nb_neighbors_spin.valueChanged.connect(self.on_parameters_changed)
        stat_layout.addRow("Number of Neighbors:", self.nb_neighbors_spin)

        self.std_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.std_ratio_spin.setRange(0.1, 10.0)
        self.std_ratio_spin.setSingleStep(0.1)
        self.std_ratio_spin.setValue(2.0)
        self.std_ratio_spin.valueChanged.connect(self.on_parameters_changed)
        stat_layout.addRow("Standard Deviation Ratio:", self.std_ratio_spin)

        stat_group.setLayout(stat_layout)
        layout.addWidget(stat_group)

        # Radius Cleaning Parameters
        radius_group = QtWidgets.QGroupBox("Radius Cleaning")
        radius_layout = QtWidgets.QFormLayout()

        self.use_radius_cb = QtWidgets.QCheckBox("Use Radius Cleaning")
        self.use_radius_cb.stateChanged.connect(self.on_parameters_changed)
        radius_layout.addRow(self.use_radius_cb)

        self.nb_points_spin = QtWidgets.QSpinBox()
        self.nb_points_spin.setRange(1, 100)
        self.nb_points_spin.setValue(16)
        self.nb_points_spin.valueChanged.connect(self.on_parameters_changed)
        radius_layout.addRow("Number of Points:", self.nb_points_spin)

        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 10.0)
        self.radius_spin.setSingleStep(0.1)
        self.radius_spin.setValue(5.0)
        self.radius_spin.valueChanged.connect(self.on_parameters_changed)
        radius_layout.addRow("Radius:", self.radius_spin)

        radius_group.setLayout(radius_layout)
        layout.addWidget(radius_group)

        # Remove Spikes Parameters
        spikes_group = QtWidgets.QGroupBox("Remove Spikes")
        spikes_layout = QtWidgets.QFormLayout()

        self.remove_spikes_cb = QtWidgets.QCheckBox("Remove Spikes")
        self.remove_spikes_cb.stateChanged.connect(self.on_parameters_changed)
        spikes_layout.addRow(self.remove_spikes_cb)

        self.iqr_multiplier_spin = QtWidgets.QDoubleSpinBox()
        self.iqr_multiplier_spin.setRange(0.1, 10.0)
        self.iqr_multiplier_spin.setSingleStep(0.1)
        self.iqr_multiplier_spin.setValue(1.5)
        self.iqr_multiplier_spin.valueChanged.connect(self.on_parameters_changed)
        spikes_layout.addRow("IQR Multiplier:", self.iqr_multiplier_spin)

        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 100)
        self.k_spin.setValue(30)
        self.k_spin.valueChanged.connect(self.on_parameters_changed)
        spikes_layout.addRow("k:", self.k_spin)

        spikes_group.setLayout(spikes_layout)
        layout.addWidget(spikes_group)

        # Remove Edge Noise Parameters
        edge_noise_group = QtWidgets.QGroupBox("Remove Edge Noise")
        edge_noise_layout = QtWidgets.QFormLayout()

        self.remove_edge_noise_cb = QtWidgets.QCheckBox("Remove Edge Noise")
        self.remove_edge_noise_cb.stateChanged.connect(self.on_parameters_changed)
        edge_noise_layout.addRow(self.remove_edge_noise_cb)

        self.curvature_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.curvature_threshold_spin.setRange(0.0, 1.0)
        self.curvature_threshold_spin.setSingleStep(0.01)
        self.curvature_threshold_spin.setValue(0.1)
        self.curvature_threshold_spin.valueChanged.connect(self.on_parameters_changed)
        edge_noise_layout.addRow("Curvature Threshold:", self.curvature_threshold_spin)

        self.k_curvature_spin = QtWidgets.QSpinBox()
        self.k_curvature_spin.setRange(1, 100)
        self.k_curvature_spin.setValue(30)
        self.k_curvature_spin.valueChanged.connect(self.on_parameters_changed)
        edge_noise_layout.addRow("k (Curvature):", self.k_curvature_spin)

        edge_noise_group.setLayout(edge_noise_layout)
        layout.addWidget(edge_noise_group)

        # Update button
        update_button = QtWidgets.QPushButton("Update")
        update_button.clicked.connect(self.on_update_clicked)
        layout.addWidget(update_button)

        layout.addStretch()

    @QtCore.Slot()
    def on_parameters_changed(self):
        # Update self.params based on UI components
        self.params['cleaning_methods']['statistical']['use_statistical'] = self.use_statistical_cb.isChecked()
        self.params['cleaning_methods']['statistical']['nb_neighbors'] = self.nb_neighbors_spin.value()
        self.params['cleaning_methods']['statistical']['std_ratio'] = self.std_ratio_spin.value()

        self.params['cleaning_methods']['radius']['use_radius'] = self.use_radius_cb.isChecked()
        self.params['cleaning_methods']['radius']['nb_points'] = self.nb_points_spin.value()
        self.params['cleaning_methods']['radius']['radius'] = self.radius_spin.value()

        self.params['remove_spikes'] = self.remove_spikes_cb.isChecked()
        self.params['iqr_multiplier'] = self.iqr_multiplier_spin.value()
        self.params['k'] = self.k_spin.value()

        self.params['remove_edge_noise'] = self.remove_edge_noise_cb.isChecked()
        self.params['curvature_threshold'] = self.curvature_threshold_spin.value()
        self.params['k_curvature'] = self.k_curvature_spin.value()

    @QtCore.Slot()
    def on_update_clicked(self):
        # Capture the current camera parameters
        # camera_params = self.visualizer.vis.get_view_control().convert_to_pinhole_camera_parameters()

        # Start the processing in a separate thread
        self.processing_thread = PointCloudProcessingThread(self.pcd, self.uvs, self.params)
        self.processing_thread.update_point_cloud.connect(self.update_point_cloud)
        self.processing_thread.start()

    @QtCore.Slot(o3d.geometry.PointCloud)
    def update_point_cloud(self, pcd):
        # Update the Open3D visualizer with the new point cloud
        self.visualizer.set_point_cloud(pcd)

    def closeEvent(self, event):
        # Clean up threads on exit
        if hasattr(self, 'processing_thread'):
            self.processing_thread.stop()
        self.visualizer.close()
        event.accept()


# Include your MeshGenerator class here or import it if it's in a separate module
# For this example, we will use a simplified MeshGenerator
class MeshGenerator:
    def __init__(self, pcd: o3d.geometry.PointCloud, uvs=None):
        """
        Initialize the MeshGenerator with an Open3D PointCloud and optional UV coordinates.

        Parameters:
            pcd (o3d.geometry.PointCloud): The point cloud to process.
            uvs (np.array): The UV coordinates for each point (N x 2).
        """
        self.pcd = pcd
        self.uvs = uvs

    def clean_point_cloud(self, method_params):
        ind = np.arange(len(self.pcd.points))
        for method_name, params in method_params.items():
            if method_name == 'statistical' and params.get('use_statistical', False):
                nb_neighbors = params.get('nb_neighbors', 20)
                std_ratio = params.get('std_ratio', 2.0)
                _, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio,
                                                             print_progress=True)
                self.pcd = self.pcd.select_by_index(ind)
            elif method_name == 'radius' and params.get('use_radius', False):
                nb_points = params.get('nb_points', 16)
                radius = params.get('radius', 0.05)
                _, ind = self.pcd.remove_radius_outlier(nb_points=nb_points, radius=radius, print_progress=True)
                self.pcd = self.pcd.select_by_index(ind)
        # Update UVs
        if self.uvs is not None:
            self.uvs = self.uvs[ind]

    def remove_spikes_dynamic(self, iqr_multiplier=1.5, k=30):
        points = np.asarray(self.pcd.points)
        tree = o3d.geometry.KDTreeFlann(self.pcd)
        distances = []
        for i in range(len(points)):
            _, idx, dist = tree.search_knn_vector_3d(points[i], k)
            distances.append(np.mean(dist))
        distances = np.array(distances)
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        threshold = q3 + iqr_multiplier * iqr
        valid_indices = distances < threshold
        self.pcd = self.pcd.select_by_index(np.where(valid_indices)[0])
        if self.uvs is not None:
            self.uvs = self.uvs[valid_indices]

    def remove_edge_noise(self, curvature_threshold=0.1, k=30):
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        curvatures = self.compute_curvature(self.pcd, k)
        valid_indices = curvatures < curvature_threshold
        self.pcd = self.pcd.select_by_index(np.where(valid_indices)[0])
        if self.uvs is not None:
            self.uvs = self.uvs[valid_indices]

    def compute_curvature(self, pcd, k):
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
    app = QtWidgets.QApplication(sys.argv)
    from nukebridge.utils.image_io import get_image_io

    imgIO = get_image_io()
    img_exr = r'C:/Users/Femto7000/geoanything/test2/GAI_PointCloudGen1/20240925/sources/depth_to_position.%04d.exr'
    points = imgIO.read_image(img_exr, frame_range=(1001, 1001), output_format='np')[0]
    size = points.shape[:2]
    uvs = generate_uvs_from_image(size)

    r, g, b, a = points[:, :, 0], points[:, :, 1], points[:, :, 2], points[:, :, 3]
    mask = a > 0  # Filter out points with alpha > 0
    # Create an array of vectors for points with alpha > 0
    points = np.vstack((r[mask], g[mask], b[mask])).T
    uvs = uvs[mask]

    # Create an Open3D PointCloud object
    initial_pcd = o3d.geometry.PointCloud()
    initial_pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = Open3DWidget(initial_pcd, parent=None)
    visualizer.show()
    window = MainWindow(initial_pcd, uvs)
    # window.show()

    sys.exit(app.exec())
