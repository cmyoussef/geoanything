import os
import nuke

from nukebridge.gizmos.core.base import Base
from geoanything.config.config_loader import ConfigLoader


class PanoramaMoGe(Base):
    def __init__(self, gizmo=None, name=None):
        self.config_loader = ConfigLoader()
        self.module_name = 'GeoAnythingPanorama'
        super().__init__(gizmo=gizmo, name=name)
        self.ext = 'exr'
        self.input_ext = 'png'
        self.create_gizmo_setup()

    def create_generate_knobs(self, *args, **kwargs):
        print('Creating knobs for PanoramaMoGe Gizmo')
        self.create_generate_tab()

        splitted_res_knob = self.create_and_configure_knob(
            'splitted_resolution_knob',
            'Int_Knob',
            'Splitted Resolution',
            default_value=512,
            flags={'STARTLINE': 'set'}
        )
        splitted_res_knob.setRange(128, 4096)

        batch_size_knob = self.create_and_configure_knob(
            'batch_size_knob',
            'Int_Knob',
            'Batch Size',
            default_value=4,
            flags={'STARTLINE': 'set'}
        )
        batch_size_knob.setRange(1, 64)

        # You could add more knobs for rotate_x_neg90, flip_v, remove_edges, etc.
        # e.g.

        switch_to_multilayer = self.create_and_configure_knob(
            'switch_to_multilayer',
            'Boolean_Knob',
            'Multi layer',
            default_value=False,
            flags={'STARTLINE': 'set'}
        )
        remove_edges_knob = self.create_and_configure_knob(
            'remove_edges_knob',
            'Boolean_Knob',
            'Remove Edges',
            default_value=True,
            flags={'STARTLINE': 'set'}
        )

        super().create_generate_knobs(*args, **kwargs)

    def create_gizmo_setup(self):
        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo.setInput(0, self.input_node)
        self.output_node.setInput(0, read_geo)

    def update_args(self, *args, **kwargs):
        """
        Collect values from the knobs and pass them to the executor
        via 'self.args_dict'.
        """
        # typical approach
        self.args['model_path'] = self.config_loader.script_paths['MoGe']

        # The user-specified .obj pattern, e.g. "C:/out/panorama_geo.%04d.obj"
        self.args['output_geo'] = os.path.join(
            self.output_path,
            f'panorama_geo.{self.frame_padding}.obj'
        ).replace('\\', '/')

        splitted_resolution = self.gizmo['splitted_resolution_knob'].value()
        self.args['splitted_resolution'] = splitted_resolution

        batch_size = self.gizmo['batch_size_knob'].value()
        self.args['batch_size'] = batch_size

        # We read the new knobs (rotate_x_neg90, flip_v, remove_edges)
        remove_edges = self.gizmo['remove_edges_knob'].value()

        self.args['remove_edges'] = remove_edges

        read_geo = self.get_node("geo_reader", "ReadGeo")
        read_geo['file'].setValue(self.args['output_geo'])

        super().update_args(*args, **kwargs)
        print("args updated successfully with new values from knobs")
    def execute_callback(self, thread):
        output = super().execute_callback(thread)
        if output:
            self.logger.info(f"Panorama geo generation completed successfully: {output}")
            output_geo = output.get('output_geo')
            self.logger.debug(f"Output geo: {output_geo}")
            # handle more logic if needed
