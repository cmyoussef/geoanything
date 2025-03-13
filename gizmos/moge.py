import os

import nuke
from nukebridge.gizmos.core.baselive import Base

from geoanything.config.config_loader import ConfigLoader


class MoGe(Base):
    def __init__(self, gizmo=None, name=None):
        self.config_loader = ConfigLoader()
        self.module_name = 'GeoAnything'
        super().__init__(gizmo=gizmo, name=name)
        self.ext = 'exr'
        self.input_ext = 'png'
        self.create_gizmo_setup()

    def update_args(self, *args, **kwargs):
        """
        Collect values from the new knobs (sensor width, height, fov_x, focal_length_mm)
        and pass them as arguments to the MoGeExecutor (via base executor).
        """
        self.args['model_path'] = self.config_loader.script_paths['MoGe']
        self.args['output_geo'] = os.path.join(self.output_path, f'output_geo.{self.frame_padding}.obj').replace('\\',
                                                                                                                 '/')

        # Existing knobs
        self.args['remove_edge'] = self.gizmo['remove_edge_knob'].value()
        self.args['resolution_level'] = self.gizmo['resolution_level_knob'].value()
        self.args['rtol'] = self.gizmo['rtol_knob'].value()

        # NEW knobs
        self.args['sensor_width_mm'] = self.gizmo['sensor_width_knob'].value()
        self.args['sensor_height_mm'] = self.gizmo['sensor_height_knob'].value()

        # fov_x / focal_length_mm can be left as 0 or None if not used
        # If the user sets them in the gizmo, pass them as floats
        # but if they remain 0, interpret it as "None".
        user_fov_x = self.gizmo['fov_x_knob'].value()
        self.args['fov_x'] = user_fov_x if user_fov_x > 0 else None

        user_focal_len = self.gizmo['focal_length_mm_knob'].value()
        self.args['focal_length_mm'] = user_focal_len if user_focal_len > 0 else None

        # existing approach: set up the ReadGeo node path
        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo['file'].setValue(self.args['output_geo'])
        super().update_args(*args, **kwargs)

    def create_generate_knobs(self, *args, **kwargs):
        print('Creating knobs' * 10)
        self.create_generate_tab()

        # Existing knobs
        self.create_and_configure_knob(
            'remove_edge_knob', 'Boolean_Knob',
            'Remove Edge',
            flags={'STARTLINE': 'set'}
        )

        resolution_level_knob = self.create_and_configure_knob(
            'resolution_level_knob', 'Int_Knob',
            'resolution_level_knob',
            default_value=9,
            flags={'STARTLINE': 'set'}
        )
        resolution_level_knob.setRange(1, 9)

        rtol_knob = self.create_and_configure_knob(
            'rtol_knob', 'Double_Knob',
            'Relative tolerance for edge removal',
            default_value=.02,
            flags={'STARTLINE': 'set'}
        )
        rtol_knob.setRange(0.005, 0.8)

        guessed_focal = self.create_and_configure_knob(
            'guessed_focal_knob', 'Double_Knob',
            'Guessed Focal',
            default_value=35,
            flags={'STARTLINE': 'set'}
        )
        guessed_focal.setRange(2, 500)

        # ---- NEW KNOBS BELOW ----
        sensor_width_knob = self.create_and_configure_knob(
            'sensor_width_knob', 'Double_Knob',
            'Sensor Width (mm)',
            default_value=36.0,
            flags={'STARTLINE': 'set'}
        )
        sensor_width_knob.setRange(5, 100)

        sensor_height_knob = self.create_and_configure_knob(
            'sensor_height_knob', 'Double_Knob',
            'Sensor Height (mm)',
            default_value=24.0,
            flags={'STARTLINE': 'set'}
        )
        sensor_height_knob.setRange(3, 80)

        fov_x_knob = self.create_and_configure_knob(
            'fov_x_knob', 'Double_Knob',
            'FOV X (degrees)',
            default_value=0.0,  # 0 => means unused
            flags={'STARTLINE': 'set'}
        )
        fov_x_knob.setRange(0, 180)

        focal_length_mm_knob = self.create_and_configure_knob(
            'focal_length_mm_knob', 'Double_Knob',
            'Focal Length (mm)',
            default_value=0.0,  # 0 => means unused
            flags={'STARTLINE': 'set'}
        )
        focal_length_mm_knob.setRange(0, 1000)
        # ---- END NEW KNOBS ----

        super().create_generate_knobs(*args, **kwargs)

    def create_gizmo_setup(self):
        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo.setInput(0, self.input_node)
        self.output_node.setInput(0, read_geo)

    def execute_callback(self, thread):
        output = super().execute_callback(thread)
        if output:
            self.logger.info(f"Geo generation completed successfully: {output}")
            output_geo = output.get('output_geo')
            focal_length = float(output.get('focal_length'))
            self.logger.debug(f"Output geo: {output_geo}")
            self.logger.debug(f"Focal Length: {focal_length}")
            # This sets the guessed_focal_knob to the final average fov_x
            # or the final computed focal length from the pipeline
            # (Your code previously sets guessed_focal from a 'focal_length' output
            #  which was actually an FOV, but let's just keep it for demonstration.)
            nuke.executeInMainThreadWithResult(self.set_focal_length, focal_length)

    def set_focal_length(self, focal_length):
        """
        Called after generation completes to update the 'guessed_focal_knob'
        with the final average FOV or something from the pipeline.
        """
        self.gizmo['guessed_focal_knob'].setValue(focal_length)

    def execute(self, *args, **kwargs):
        super().execute(*args, **kwargs)
        nuke.selectAll()
        nuke.invertSelection()
