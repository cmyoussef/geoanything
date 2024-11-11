import os

import nuke
from nukebridge.gizmos.core.baselive import BaseLive, Icons, SocketClient

from geoanything.config.config_loader import ConfigLoader


class PointCloudGen(BaseLive):
    def __init__(self, gizmo=None, name=None):
        self.config_loader = ConfigLoader()
        self.module_name = 'GeoAnything'
        super().__init__(gizmo=gizmo, name=name)
        self.ext = 'exr'
        self.input_ext = 'png'


    def create_generate_knobs(self):
        """
        Sets up the UI knobs under the 'Generate' tab.
        This method is responsible for creating and organizing knobs related to the generation process.
        """
        self.create_generate_tab()

        self.create_terminal_execution_knobs()

        self.add_divider()

        self.create_frame_range_knobs()
        self.set_status(running=False)

    def create_terminal_execution_knobs(self):
        """
        Creates knobs for controlling the execution of processes in an external terminal.
        This includes options to execute, interrupt, and manage execution settings.
        """
        # PyScript knob for execution with a command
        self.create_and_configure_knob(
            'execute_btn', 'PyScript_Knob', f'Generate PointCloud {Icons.execute_symbol}',
            flags={'STARTLINE': 'set'},
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_execute")'
        )
        self.execute_btn_list.append('execute_btn')

        self.create_gizmo_setup()

        self.create_meshing_knobs()

        self.create_and_configure_knob('geo_view', 'Boolean_Knob',
                                       f'geo_view{Icons.detect_image_size_symbol}',
                                       flags={'STARTLINE': 'set'}
                                       )
        self.create_and_configure_knob('use_input_mask_knob', 'Boolean_Knob',
                                       f'Use Input mask',
                                       # flags={'STARTLINE': 'set'}
                                       )

        # PyScript knob for execution with a command
        self.create_and_configure_knob(
            'execute_geo_btn', 'PyScript_Knob', f'Generate Mesh {Icons.execute_symbol}',
            flags={'STARTLINE': 'set'},
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_geo_execute")'
        )
        self.execute_btn_list.append('execute_geo_btn')
        # Reload button
        self.reload_button()

        # PyScript knob for interrupting an operation with a command
        self.create_and_configure_knob(
            'interrupt_btn', 'PyScript_Knob', f'Interrupt {Icons.interrupt_symbol}',
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_interrupt")'
        )

        # region terminal execution
        self.create_and_configure_knob(
            'use_external_execute', 'Boolean_Knob', f'Use Farm {Icons.tractor_symbol}',
            flags={'STARTLINE': 'set'}
        )

        check_box_knob = self.gizmo.knob('geo_view')
        switch_node = self.get_node('Switch_vis')
        switch_node['which'].setExpression(f"{self.gizmo.name()}.{check_box_knob.name()}")

        # Create a MergeExpression node to merge the input and output alpha channels
        input_expression_node = self.get_node("input_alpha_exp", "MergeExpression")
        use_input_mask_knob = self.gizmo.knob('use_input_mask_knob')
        input_expression_node['disable'].setExpression(f"1-{self.gizmo.name()}.{use_input_mask_knob.name()}")
        # endregion

    def estimate_focal(self):
        self.update_args()
        self.args['script_path'] = self.config_loader.script_paths['FocalLengthEstimator']
        self.args['frame_range'] = self.frame_range
        self.args['sensor_width_mm'] = self.gizmo.knob('sensor_width_mm_knob').value()
        self.execute(update_output=False, callback=self.estimate_focal_callback)

    def estimate_focal_callback(self, thread):
        output = self.execute_callback(thread)
        nuke.executeInMainThreadWithResult(self._estimate_focal_callback, (output, ))


    def _estimate_focal_callback(self, output):
        self.logger.info(f"Focal length estimation completed successfully: {output}")
        focal = output.get('focal_length_mm', 36)
        self.gizmo['focal'].setValue(focal)

    def on_execute(self, silent=False):
        self.args['script_path'] = self.config_loader.script_paths['PointCloudGen']
        self.args.pop('output_geo', None)
        super().on_execute(silent)

    def on_geo_execute(self):
        """
        Executes the geo generation process with updated parameters from the UI.
        """
        self.export_env()

        # Update arguments with logger level, Python executable, and cache directory
        self.args['logger_level'] = self.logger.get_level_names().get(self.gizmo.knob('logger_level_menu').value(), 20)
        self.args['cache_dir'] = self.cache_dir
        self.args['script_path'] = self.config_loader.script_paths['PointToGeo']
        self.args['frame_range'] = self.frame_range

        os.makedirs(self.output_path, exist_ok=True)

        # Retrieve the depth-to-position image path
        depth_to_position_node = self.get_node('depth_to_position')
        depth_to_position_path = self.get_input_img_path(depth_to_position_node)
        self.args['output_geo'] = os.path.join(self.output_path, f'output_geo.{self.frame_padding}.obj').replace('\\',
                                                                                                                 '/')
        depth_to_position_path = self.write_input(depth_to_position_path, node=depth_to_position_node, ext='exr',
                                                  frame_range=None, temp=False, hard_error=False)
        self.write_inputs()
        self.args['inputs']['colors'] = self.args['inputs'][list(self.args['inputs'].keys())[0]]
        self.args['inputs']['positions'] = depth_to_position_path

        # Collect mesh generation and cleaning parameters
        self.args['data'] = self.get_cleanup_params()

        # Log the execution details
        self.logger.info(f"Running {self.__class__.__name__}.on_geo_execute with arguments: {self.args}")

        # Update the file path for the ReadGeo node
        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo['file'].setValue(self.args['output_geo'])

        # Execute with the updated arguments
        self.execute(update_ouput=False, callback=self.on_geo_execute_callback)

    def on_geo_execute_callback(self, thread):
        """
        Callback method for the geo generation process.
        """
        output = self.execute_callback(thread)
        if output:
            self.logger.info(f"Geo generation completed successfully: {output}")
            self.gizmo['geo_view'].setValue(True)
            read_geo = self.get_node("geo_reader", "ReadGeo")
            read_geo['reload'].execute()
            # self.reload_read_nodes()

    def reload_read_nodes(self):
        """
        Reloads all Read nodes within the gizmo. This is typically used to refresh the content of Read nodes
        after changes to their source files.

        """

        for n in self.gizmo.nodes():
            # Check if the current node is a Read node
            if n.Class() in ['Read', 'ReadGeo']:
                # Reload the Read node
                n['reload'].execute()
        self.force_evaluate_nodes()

    def update_single_read_node(self, node, file_path):
        # Update the file path and set the colorspace to linear
        super().update_single_read_node(node, file_path)
        node.knob("colorspace").setValue("linear")

    def create_gizmo_setup(self):
        self.add_divider('Point Cloud Edit')
        nuke.selectAll()
        nuke.invertSelection()
        read_node = self.get_node('Read1', 'Read')
        read_node['raw'].setValue(True)

        # Create a Grade node, connect it, and set properties
        grade_node = self.get_node("depth_grade", "Grade")
        grade_node.setInput(0, read_node)
        grade_node['add'].setValue(1)
        grade_node['multiply'].setValue(1)
        grade_node['gamma'].setValue(1)
        for knob in ['gamma', 'add', 'multiply']:
            self.clone_and_link_knob(grade_node[knob])

        # Create a Copy node and set connections
        copy_node = self.get_node("depth_copy", "Copy")
        copy_node.setInput(1, grade_node)
        copy_node.setInput(0, self.input_node)
        copy_node['from0'].setValue("r")
        copy_node['to0'].setValue("depth.Z")

        # Create an Expression node, connect it, and set expressions
        expression_node = self.get_node("alpha_exp", "Expression")
        expression_node.setInput(0, read_node)
        expression_node['expr0'].setValue("clamp(r*100000, 0, 1)")
        expression_node['channel0'].setValue("alpha")

        nuke.selectAll()
        nuke.invertSelection()

        # Create an Expression node, connect it, and set expressions
        input_expression_node = self.get_node("input_alpha_exp", "MergeExpression")
        input_expression_node.setInput(0, expression_node)
        input_expression_node.setInput(1, self.input_node)
        input_expression_node['expr0'].setValue("clamp(Aa*10000, 0, 1) * Ba")
        input_expression_node['channel0'].setValue("alpha")

        nuke.selectAll()
        nuke.invertSelection()
        # Create another Copy node to copy the altered alpha to the output of the Grade node
        copy_alpha_node = self.get_node('alpha_copy', "Copy")
        copy_alpha_node.setInput(0, copy_node)
        copy_alpha_node.setInput(1, input_expression_node)
        copy_alpha_node['from0'].setValue("alpha")
        copy_alpha_node['to0'].setValue("alpha")

        nuke.selectAll()
        nuke.invertSelection()

        # Create a DepthToPoints node and set properties
        depth_to_position = self.get_node("depth_to_position", "DepthToPosition")
        depth_to_position.setInput(0, copy_alpha_node)
        self.add_divider('Camera')
        # Create a Camera node and connect it

        self.create_and_configure_knob('sensor_width_mm_knob', 'Double_Knob', 'Sensor Width mm',
                                       default_value=36.0,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob(
            'focal_estimate_btn', 'PyScript_Knob', f'Guss Focal {Icons.execute_symbol}',
            flags={'STARTLINE': 'set'},
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("estimate_focal")'
        )
        self.execute_btn_list.append('focal_estimate_btn')

        self.camera_input = self.get_node('userCamera', 'Input')
        camera_node = self.get_node("camera_vis", "Camera4")
        camera_node['focal'].setValue(75)
        camera_node['haperture'].setValue(50)
        for knob in ['focal', 'haperture', 'vaperture', 'translate', 'rotate', 'scaling', 'uniform_scale']:
            self.clone_and_link_knob(camera_node[knob])

        # Create a Switch node
        switch_node = self.get_node("user_camera_switch", "Switch")

        switch_node.setInput(0, camera_node)
        switch_node.setInput(1, self.camera_input)
        switch_node['which'].setExpression(f"parent.inputs == 2")

        depth_to_position.setInput(1, switch_node)

        position_to_points = self.get_node("position_vis", "PositionToPoints")
        position_to_points.setInput(0, depth_to_position)
        position_to_points.setInput(1, copy_alpha_node)

        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo.setInput(0, self.input_node)

        # Create a Switch node
        switch_node = self.get_node("Switch_vis", "Switch")
        switch_node.setInput(1, read_geo)
        switch_node.setInput(0, position_to_points)
        self.output_node.setInput(0, switch_node)

    def create_meshing_knobs(self):
        """
        Creates knobs for controlling the meshing process and cleaning point cloud.
        This includes options for various cleaning methods and mesh generation parameters.
        """
        # region meshing
        # self.add_divider('Meshing')

        # Cleaning Methods
        self.add_divider('Cleaning Methods')

        # Statistical Outlier Removal
        self.create_and_configure_knob('use_statistical', 'Boolean_Knob', 'Use Statistical Cleaning',
                                       default_value=False,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('nb_neighbors', 'Int_Knob', 'Number of Neighbors',
                                       default_value=30,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('std_ratio', 'Double_Knob', 'Standard Deviation Ratio',
                                       default_value=2.0,
                                       flags={'STARTLINE': 'set'}
                                       )

        # Radius Outlier Removal
        self.create_and_configure_knob('use_radius', 'Boolean_Knob', 'Use Radius Cleaning',
                                       default_value=True,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('nb_points', 'Int_Knob', 'Number of Points',
                                       default_value=16,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('radius', 'Double_Knob', 'Radius',
                                       default_value=5.0,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.add_divider('Spike and Edge Removal')

        # Remove Spikes Dynamic
        self.create_and_configure_knob('remove_spikes', 'Boolean_Knob', 'Remove Spikes',
                                       default_value=False,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('iqr_multiplier', 'Double_Knob', 'IQR Multiplier',
                                       default_value=1.5,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('k', 'Int_Knob', 'K Neighbors for Spikes',
                                       default_value=30,
                                       flags={'STARTLINE': 'set'}
                                       )

        # Remove Edge Noise
        self.create_and_configure_knob('remove_edge_noise', 'Boolean_Knob', 'Remove Edge Noise',
                                       default_value=False,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('curvature_threshold', 'Double_Knob', 'Curvature Threshold',
                                       default_value=0.1,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('k_curvature', 'Int_Knob', 'K Neighbors for Curvature',
                                       default_value=30,
                                       flags={'STARTLINE': 'set'}
                                       )

        # Mesh Generation Parameters
        self.add_divider('Mesh Generation Parameters')
        # Logger level selection
        self.create_and_configure_knob(
            'meshing_method_menu', 'Enumeration_Knob', 'Meshing Method',
            flags={'STARTLINE': 'set'},
            values=['poisson', 'bpa'],
            default_value='INFO'
        )

        self.create_and_configure_knob('depth', 'Int_Knob', 'Octree Depth',
                                       default_value=9,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('width', 'Int_Knob', 'Grid Width',
                                       default_value=0,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('scale', 'Double_Knob', 'Scale',
                                       default_value=1.0,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('linear_fit', 'Boolean_Knob', 'Use Linear Fit',
                                       default_value=True,
                                       flags={'STARTLINE': 'set'}
                                       )

        # Density Threshold
        self.create_and_configure_knob('density_threshold', 'Double_Knob', 'Density Threshold',
                                       default_value=0.0,
                                       flags={'STARTLINE': 'set'}
                                       )

        # Remove Holes
        self.create_and_configure_knob('remove_holes', 'Boolean_Knob', 'Remove Holes',
                                       default_value=True,
                                       flags={'STARTLINE': 'set'}
                                       )

        # Number of Threads (optional)
        self.create_and_configure_knob('n_threads', 'Int_Knob', 'Number of Threads',
                                       default_value=-1,
                                       flags={'STARTLINE': 'set'}
                                       )

        # Add a button to launch the visualizer
        self.create_and_configure_knob(
            'launch_visualizer_btn', 'PyScript_Knob', 'Launch Visualizer',
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_launch_visualizer")'
        )

        # Add an update button to send parameters to the visualizer
        self.create_and_configure_knob(
            'update_visualizer_btn', 'PyScript_Knob', 'Update Visualizer',
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_update_visualizer")'
        )
        # endregion

    def on_launch_visualizer(self):
        """
        Launches the visualizer application with the necessary arguments.
        """
        self.logger.info("Launching visualizer...")
        self.update_args()
        # Prepare arguments
        self.args['script_path'] = self.config_loader.script_paths['PointCloudVisualizer']
        self.args['port'] = SocketClient.find_available_port()
        self.visualizer_port = self.args['port']  # Store the port for client connection

        # Retrieve the pre and post commands from the gizmo's knobs
        pre_cmd = self.gizmo.knob('pre_cmd_knob').value() or None
        post_cmd = self.gizmo.knob('post_cmd_knob').value() or None
        # open_new_terminal = self.gizmo.knob('open_new_terminal').value() or None
        # Get the initial point cloud data
        depth_to_position_node = self.get_node('depth_to_position')
        depth_to_position_path = self.get_input_img_path(depth_to_position_node)
        positions_path = self.write_input(depth_to_position_path, node=depth_to_position_node, ext='exr',
                                          frame_range=None, temp=False, hard_error=False)
        self.write_inputs()

        self.args['positions'] = positions_path
        self.args['colors'] = self.args['inputs'][list(self.args['inputs'].keys())[0]]

        # Launch the visualizer using ExecuteThreadManager
        thread = self.thread_manager.get_thread(
            args=self.args,
            # python_exe=self.python_exe,
            script_path=self.args['script_path'],
            pre_cmd=pre_cmd,
            post_cmd=post_cmd,
            export_env=self.export_env(),
            open_new_terminal=False
        )
        thread.daemon = True
        thread.start()
        self.visualizer_thread = thread
        self.THREADS_LIST.append(thread)
        printed_cmd = thread.cmd.rsplit('--encoded-args', 1)[0]

        nuke.tprint(f"{'-' * 100}\nExecute {self.__class__.__name__}:\n{printed_cmd}\n{self.args}\n{'-' * 100}")

        # Create the client to communicate with the visualizer
        self.visualizer_client = SocketClient(host='localhost', port=self.visualizer_port)
        connected = self.visualizer_client.attempting_to_connect(max_retries=10)
        # if connected:
        #     self.logger.info("Connected to the visualizer server.")
        # else:
        #     self.logger.error("Failed to connect to the visualizer server.")

    def get_cleanup_params(self):
        # Collect parameters from the gizmo knobs
        params = {
            'cleaning_methods': {
                'statistical': {
                    'use_statistical': self.gizmo.knob('use_statistical').value(),
                    'nb_neighbors': int(self.gizmo.knob('nb_neighbors').value()),
                    'std_ratio': float(self.gizmo.knob('std_ratio').value())
                },
                'radius': {
                    'use_radius': self.gizmo.knob('use_radius').value(),
                    'nb_points': int(self.gizmo.knob('nb_points').value()),
                    'radius': float(self.gizmo.knob('radius').value())
                }
            },
            'remove_spikes': self.gizmo.knob('remove_spikes').value(),
            'iqr_multiplier': float(self.gizmo.knob('iqr_multiplier').value()),
            'k': int(self.gizmo.knob('k').value()),
            'remove_edge_noise': self.gizmo.knob('remove_edge_noise').value(),
            'curvature_threshold': float(self.gizmo.knob('curvature_threshold').value()),
            'k_curvature': int(self.gizmo.knob('k_curvature').value()),

            'mesh_params': {
                'method': self.gizmo.knob('meshing_method_menu').value(),
                'depth': int(self.gizmo.knob('depth').value()),
                'width': float(self.gizmo.knob('width').value()),
                'scale': float(self.gizmo.knob('scale').value()),
                'linear_fit': self.gizmo.knob('linear_fit').value(),
                'density_threshold': float(self.gizmo.knob('density_threshold').value()),
                'remove_holes': self.gizmo.knob('remove_holes').value(),
                'n_threads': int(self.gizmo.knob('n_threads').value())
            }
        }
        return params

    def on_update_visualizer(self):
        """
        Collects parameters from the gizmo and sends them to the visualizer via the client.
        """
        self.logger.info("Updating visualizer...")

        params = self.get_cleanup_params()

        # Send parameters to the visualizer via the client
        if hasattr(self, 'visualizer_client') and self.visualizer_client.is_client_connected:
            response = self.visualizer_client.send_data_and_wait_for_response(params)
            self.logger.info(f"Visualizer response: {response}")
        else:
            self.logger.error("Client is not connected to the visualizer.")

    def on_destroy(self):
        """
        Cleans up resources when the gizmo is destroyed.
        """
        super().on_destroy()
        if hasattr(self, 'visualizer_thread'):
            self.visualizer_thread.terminate()
            self.logger.info("Visualizer process terminated.")
        if hasattr(self, 'visualizer_client'):
            self.visualizer_client.disconnect_from_server()
            self.logger.info("Disconnected from visualizer server.")
