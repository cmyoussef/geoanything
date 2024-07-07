import nuke
from nukebridge.gizmos.core.base import Base, Icons

from geoanything.config.config_loader import ConfigLoader


class PointCloudGen(Base):
    def __init__(self, gizmo=None, name=None):
        self.config_loader = ConfigLoader()
        self.module_name = 'GeoAnything'
        super().__init__(gizmo=gizmo, name=name)

        self.ext = 'png'

    def populate_ui(self):
        self.create_generate_tab()
        self.create_gizmo_setup()

        super().populate_ui()

    def create_terminal_execution_knobs(self):
        """
        Creates knobs for controlling the execution of processes in an external terminal.
        This includes options to execute, interrupt, and manage execution settings.
        """
        # region terminal execution
        self.create_and_configure_knob(
            'use_external_execute', 'Boolean_Knob', f'Use Farm {Icons.tractor_symbol}',
            flags={'STARTLINE': 'set'}
        )

        # PyScript knob for execution with a command
        self.create_and_configure_knob(
            'execute_btn', 'PyScript_Knob', f'Generate PointCloud {Icons.execute_symbol}',
            flags={'STARTLINE': 'set'},
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_execute")'
        )

        # PyScript knob for execution with a command
        self.create_and_configure_knob(
            'execute_geo_btn', 'PyScript_Knob', f'Generate Mesh {Icons.execute_symbol}',
            # flags={'STARTLINE': 'set'},
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_execute")'
        )
        # Reload button
        self.reload_button()

        # PyScript knob for interrupting an operation with a command
        self.create_and_configure_knob(
            'interrupt_btn', 'PyScript_Knob', f'Interrupt {Icons.interrupt_symbol}',
            command=f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_interrupt")'
        )
        # endregion

    def update_single_read_node(self, node, file_path):
        # Update the file path and set the colorspace to linear
        super().update_single_read_node(node, file_path)
        node.knob("colorspace").setValue("linear")

    def create_gizmo_setup(self):
        self.add_divider('Grade')
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

        # Create another Copy node to copy the altered alpha to the output of the Grade node
        copy_alpha_node = self.get_node('alpha_copy', "Copy")
        copy_alpha_node.setInput(0, copy_node)
        copy_alpha_node.setInput(1, expression_node)
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

        # Create a Switch node
        switch_node = self.get_node("Switch_vis", "Switch")
        switch_node.setInput(0, depth_to_position)
        switch_node.setInput(1, position_to_points)
        self.output_node.setInput(0, switch_node)

        check_box_knob = self.create_and_configure_knob('3d_view', 'Boolean_Knob',
                                                        f'3d_view{Icons.detect_image_size_symbol}',
                                                        flags={'STARTLINE': 'set'})
        switch_node['which'].setExpression(f"{self.gizmo.name()}.{check_box_knob.name()}")
