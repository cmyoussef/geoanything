import sys
import threading

paths = [r'E:\stable-diffusion\stable-diffusion-integrator', r'E:\track_anything_project',
         r'E:\ai_projects\dust3r_project\vision-forge', r'E:\nuke-bridge', 'E:/ai_projects/ai_portal']
for p in paths:
    if not p in sys.path:
        sys.path.append(p)

import numpy as np
import open3d as o3d
from PyServerManager.server_manager.server import SocketServer  # Adjust import based on your project structure
from PySide6 import QtWidgets, QtCore
from nukebridge.executors.core.baseliveexecutor import BaseLiveExecutor
from geoanything.executors.utils.mesh_generator import MeshGenerator


class Open3DWidget(QtWidgets.QWidget):
    def __init__(self, point_cloud, parent=None):
        super(Open3DWidget, self).__init__(parent)
        self.point_cloud = point_cloud
        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(visible=True)
        render_option = self.vis.get_render_option()
        render_option.point_size = 5.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.line_width = 2.0
        render_option.show_coordinate_frame = True

        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_up([0.0, -1.0, 0.0])

        self.img = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(100)

        self.vis.add_geometry(self.point_cloud)

    def update_visualization(self):
        if self.point_cloud is not None:
            self.vis.update_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def set_point_cloud(self, point_cloud):
        """Update the point cloud data without clearing the geometry."""
        if self.point_cloud is not None:
            self.point_cloud.points = point_cloud.points
            if point_cloud.has_colors():
                self.point_cloud.colors = point_cloud.colors
            else:
                self.point_cloud.colors = None
            self.vis.update_geometry(self.point_cloud)
        else:
            self.point_cloud = point_cloud
            self.vis.add_geometry(self.point_cloud)

    def closeEvent(self, event):
        self.vis.destroy_window()
        event.accept()


class PointCloudVisualizer(QtWidgets.QWidget):
    parameters_received = QtCore.Signal(dict)

    def __init__(self, mesh_generator, parent=None):
        super().__init__(parent)
        self.mesh_generator = mesh_generator  # Store the MeshGenerator instance

        # Use the point cloud from the mesh_generator
        self.pcd = self.mesh_generator.pcd

        # Set up the Open3DWidget
        self.visualizer = Open3DWidget(self.pcd, parent=self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.visualizer)
        self.setLayout(layout)

        # Connect the signal for updating the point cloud
        self.parameters_received.connect(self.update_point_cloud)

    @QtCore.Slot(dict)
    def update_point_cloud(self, params):
        # Reset the point cloud in the mesh_generator
        self.mesh_generator.reset_pcd()

        # Process cleaning methods
        cleaning_methods = params.get('cleaning_methods', {})
        if cleaning_methods:
            self.mesh_generator.clean_point_cloud(cleaning_methods)

        # Remove spikes
        if params.get('remove_spikes', False):
            iqr_multiplier = params.get('iqr_multiplier', 1.5)
            k = params.get('k', 30)
            self.mesh_generator.remove_spikes_dynamic(iqr_multiplier=iqr_multiplier, k=k)

        # Remove edge noise
        if params.get('remove_edge_noise', False):
            curvature_threshold = params.get('curvature_threshold', 0.1)
            k_curvature = params.get('k_curvature', 30)
            self.mesh_generator.remove_edge_noise(curvature_threshold=curvature_threshold, k=k_curvature)

        # Update the point cloud in the visualizer
        self.visualizer.set_point_cloud(self.mesh_generator.pcd)

    def closeEvent(self, event):
        # Close the visualizer window
        self.visualizer.close()
        event.accept()

class VisualizerExecutor(BaseLiveExecutor):
    def __init__(self):
        # Initialize the argument parser and process arguments
        super().__init__(skip_setup_parser=True)
        self.setup_parser()
        self.processes_parser()

        # Initialize the GUI application
        self.app = QtWidgets.QApplication(sys.argv)

        # Load the initial MeshGenerator instance
        self.mesh_generator = self.load_initial_point_cloud()

        # Initialize the visualizer widget
        self.visualizer_widget = PointCloudVisualizer(self.mesh_generator)
        self.visualizer_widget.show()

        # Start the server in a separate thread
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Start the Qt event loop
        sys.exit(self.app.exec())

    def setup_parser(self):
        self.parser.add_argument('--encoded-args', help='Base64 encoded JSON arguments')

    def start_server(self):
        # Initialize the server
        self.port = self.args_dict.pop('port')
        self.server = SocketServer(port=self.port, data_handler=self.live_run)
        self.server.start_accepting_clients(return_response_data=True)
        self.logger.info(f"Server started on port {self.port}")

    def live_run(self, data):
        # This method is called when data is received from the client
        # Emit the signal to update the point cloud in the GUI
        self.visualizer_widget.parameters_received.emit(data)
        return "Parameters received and processing started."

    def load_initial_point_cloud(self):
        # Load the positions image
        positions_path = self.args_dict['positions']
        frame_range = self.args_dict.get('frame_range', (1001, 1001))
        positions = self.imageIO.read_image(positions_path, frame_range=frame_range, output_format='np')[0]

        # Load the colors image if provided
        colors = None
        colors_path = self.args_dict.get('colors', None)
        if colors_path:
            colors = self.imageIO.read_image(colors_path, frame_range=frame_range, output_format='np')[0]

        # Create MeshGenerator instance using from_np class method
        mesh_generator = MeshGenerator.from_np(positions, colors=colors)

        return mesh_generator


if __name__ == '__main__':
    # Create a VisualizerExecutor instance
    executor = VisualizerExecutor()
    print('*' * 50, "Server should be running", '*' * 50)
    print(executor.args_dict)
    print('*' * 50, "************************", '*' * 50)
    try:
        lvl = int(executor.args_dict.get('logger_level', 20))
    except TypeError:
        lvl = 20
    executor.logger.setLevel(lvl)
