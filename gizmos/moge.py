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
        self.args['model_path'] = self.config_loader.script_paths['MoGe']
        self.args['output_geo'] = os.path.join(self.output_path, f'output_geo.{self.frame_padding}.obj').replace('\\',
                                                                                                                 '/')
        self.args['remove_edge'] = self.gizmo['remove_edge_knob'].value()
        self.args['resolution_level'] = self.gizmo['resolution_level_knob'].value()
        self.args['rtol'] = self.gizmo['rtol_knob'].value()
        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo['file'].setValue(self.args['output_geo'])
        super().update_args(*args, **kwargs)

    def create_generate_knobs(self, *args, **kwargs):
        print('Creating knobs' * 10)
        self.create_generate_tab()

        self.create_and_configure_knob('remove_edge_knob', 'Boolean_Knob',
                                       f'Remove Edge',
                                       flags={'STARTLINE': 'set'}
                                       )

        resolution_level_knob = self.create_and_configure_knob('resolution_level_knob', 'Int_Knob',
                                                               'resolution_level_knob',
                                                               default_value=9,
                                                               flags={'STARTLINE': 'set'}
                                                               )
        resolution_level_knob.setRange(1, 9)

        rtol_knob = self.create_and_configure_knob('rtol_knob', 'Double_Knob', 'Relative tolerance for edge removal',
                                                   default_value=.02,
                                                   flags={'STARTLINE': 'set'}
                                                   )
        rtol_knob.setRange(0.005, 0.8)


        guessed_focal = self.create_and_configure_knob('guessed_focal_knob', 'Double_Knob', 'Guessed Focal',
                                                   default_value=35,
                                                   flags={'STARTLINE': 'set'}
                                                   )
        guessed_focal.setRange(2, 500)

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
            nuke.executeInMainThreadWithResult(self.set_focal_length, focal_length)

    def set_focal_length(self, focal_length):
        self.gizmo['guessed_focal_knob'].setValue(focal_length)
