import os

import nuke
from nukebridge.gizmos.core.base import Base, Icons

from geoanything.config.config_loader import ConfigLoader


class PointCloudGen(Base):
    def __init__(self, gizmo=None, name=None):
        self.config_loader = ConfigLoader()
        self.module_name = 'GeoAnything'
        super().__init__(gizmo=gizmo, name=name)

        self.ext = 'png'

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

    def on_execute(self, silent=False):
        self.args['script_path'] = self.config_loader.script_paths['PointCloudGen']
        self.args.pop('output_geo', None)
        super().on_execute(silent)

    def on_geo_execute(self):
        self.export_env()

        self.args['logger_level'] = self.logger.logger_level.get(self.gizmo.knob('logger_level_menu').value(), 20)
        self.args['python_exe'] = self.python_exe
        self.args['cache_dir'] = self.cache_dir
        self.args['script_path'] = self.config_loader.script_paths['PointToGeo']
        self.args['frame_range'] = f'{self.frame_range}'

        os.makedirs(self.output_path, exist_ok=True)

        depth_to_position_node = self.get_node('depth_to_position')
        depth_to_position_path = self.get_input_img_path(depth_to_position_node)
        self.args['output_geo'] = os.path.join(self.output_path, f'output_geo.{self.frame_padding}.obj').replace('\\',
                                                                                                                 '/')
        depth_to_position_path = self.write_input(depth_to_position_path, node=depth_to_position_node, ext='exr',
                                                  frame_range=None, temp=False, hard_error=False)
        self.args['inputs'] = {'positions': depth_to_position_path}
        self.args['data'] = {
            'remove_spikes': self.gizmo.knob('remove_spikes').value(),
            'iqr_multiplier': self.gizmo.knob('iqr_multiplier').value(),
            'k': self.gizmo.knob('neighbors_size').value(),
            'enhanced_cleaning': self.gizmo.knob('enhanced_cleaning').value(),
            'min_samples': self.gizmo.knob('min_samples').value(),
            'eps': self.gizmo.knob('eps').value()
        }
        self.logger.info(f"Running {self.__class__.__name__}.on_geo_execute with arguments: {self.args}")
        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo['file'].setValue(self.args['output_geo'])
        self.execute(update_ouput=False)

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

        # Create a Grade node, connect it, and set properties
        grade_node = self.get_node("depth_grade", "Grade")
        grade_node.setInput(0, read_node)
        grade_node['add'].setValue(1)
        grade_node['multiply'].setValue(20)
        grade_node['gamma'].setValue(4)
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
        camera_node = self.get_node("camera_vis", "Camera")
        camera_node['focal'].setValue(75)
        camera_node['haperture'].setValue(50)
        for knob in ['focal', 'haperture', 'vaperture', 'translate', 'rotate', 'scaling', 'uniform_scale']:
            self.clone_and_link_knob(camera_node[knob])

        depth_to_position.setInput(1, camera_node)

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
        Creates knobs for controlling the meshing process.
        """
        # region meshing
        self.add_divider('Meshing')

        self.create_and_configure_knob('remove_spikes', 'Boolean_Knob', 'Remove Spikes',
                                       default_value=True,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('iqr_multiplier', 'Double_Knob', 'Voxel Size',
                                       default_value=1.5,
                                       # flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('neighbors_size', 'Int_Knob', 'Neighbors Size',
                                       default_value=30,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('enhanced_cleaning', 'Boolean_Knob', 'Cleaning geo',
                                       default_value=True,
                                       flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('min_samples', 'Int_Knob', 'Min cluster Points',
                                       default_value=10,
                                       # flags={'STARTLINE': 'set'}
                                       )

        self.create_and_configure_knob('eps', 'Double_Knob', 'neighbors max distance',
                                       default_value=0.1,
                                       flags={'STARTLINE': 'set'}
                                       )
        # endregion
