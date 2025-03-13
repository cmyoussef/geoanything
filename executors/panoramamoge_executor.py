import os
import sys
import numpy as np
import sys

paths = [
    r'E:\stable-diffusion\stable-diffusion-integrator',
    r'E:\track_anything_project',
    r'E:\ai_projects\dust3r_project\vision-forge',
    r'E:\nuke-bridge',
    'E:/ai_projects/ai_portal'
]
for p in paths:
    if p not in sys.path:
        sys.path.append(p)
from nukebridge.executors.core.baseexecutor import BaseExecutor
from geoanything.executors.external.mogewraper_panorama import MoGePanoramaInferenceRefactored
from nukebridge.utils.image_io import get_image_io


class PanoramaMoGeExecutor(BaseExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imageIO = get_image_io()

    def setup_parser(self):
        super().setup_parser()
        # (If you want any CLI arguments, define them here.)
        # In your environment, you're typically passing everything via base64-encoded JSON.

    def run(self):
        """
        This method is invoked after the base executor parses arguments.
        We'll read the user knobs from self.args_dict, run the panorama pipeline
        for each frame, and store the final paths in 'send_output'.
        """
        self.logger.info(f"Running {self.__class__.__name__}")

        # 1) parse the model path
        model_path = f'{self.args_dict.get("cache_dir")}/model.pt'
        if not model_path or not os.path.exists(model_path):
            self.logger.error(f"Invalid or missing model: {model_path}")
            return

        # 2) parse input & output patterns
        input_filename_pattern = self.args_dict['inputs']['Input1']
        output_filename_pattern = self.args_dict['output_geo']  # e.g. "C:/out/panorama_geo.%04d.obj"
        # note that we'll produce EXR with a similar naming but .exr replaced

        # 3) other user knobs
        splitted_resolution = int(self.args_dict.get('splitted_resolution', 512))
        batch_size = int(self.args_dict.get('batch_size', 4))
        device = self.args_dict.get('device', 'cuda')
        # if you have more advanced knobs (like rotate_x_neg90, flip_v, etc.), parse them here
        rotate_x_neg90 = bool(self.args_dict.get('rotate_x_neg90', True))
        flip_v = bool(self.args_dict.get('flip_v', True))
        remove_edges = bool(self.args_dict.get('remove_edges', True))
        resolution_level = int(self.args_dict.get('resolution_level', 9))
        threshold = float(self.args_dict.get('threshold', 0.03))

        # 4) figure out root folder from the output pattern
        # e.g. "C:/out/panorama_geo.%04d.obj" => root folder "C:/out"
        root_folder = os.path.dirname(output_filename_pattern)
        os.makedirs(root_folder, exist_ok=True)

        # 5) read frames
        frame_range = range(self.frame_range[0], self.frame_range[-1] + 1)
        # This returns a list of actual image paths for each frame
        image_paths = self.imageIO.parse_image_path(
            input_filename_pattern,
            frame_range=self.frame_range,
        )

        # 6) Create the panorama inference object
        panorama_inference = MoGePanoramaInferenceRefactored(
            model_path=model_path,
            root_folder=root_folder,  # we pass the base directory
            device=device,
        )

        final_obj_path = None

        # 7) Loop over frames
        for img_path, frame_number in self.logger.progress(zip(image_paths, frame_range), desc="Processing frames:"):
            # We'll produce:
            #   exr => e.g. "C:/out/panorama_geo.0001.exr"
            #   obj => e.g. "C:/out/panorama_geo.0001.obj"
            # so we do the usual pattern substitution
            exr_filename = output_filename_pattern % frame_number  # e.g. "C:/out/panorama_geo.0001.obj"
            # Replace .obj with .exr
            exr_filename = exr_filename.rsplit('.', 1)[0] + '.exr'

            obj_filename = output_filename_pattern % frame_number  # "C:/out/panorama_geo.0001.obj"
            if not obj_filename.endswith('.obj'):
                obj_filename += '.obj'

            self.logger.info(f"Frame {frame_number} => Inference on {img_path}")
            # 8) Call the panorama pipeline
            panorama_inference.infer_panorama(
                input_image_path=img_path,
                output_exr_name=os.path.basename(exr_filename),
                output_obj_name=os.path.basename(obj_filename),
                resize_to=None,              # or user knob
                resolution_level=resolution_level,
                threshold=threshold,
                batch_size=batch_size,
                remove_edges=remove_edges,
                rotate_x_neg90=rotate_x_neg90,
                flip_v=flip_v
            )

            final_obj_path = obj_filename

        # 9) send output
        self.send_output({
            'output_geo': output_filename_pattern,  # pattern
            'final_obj_path': final_obj_path or ''
        })

if __name__ == '__main__':
    executor = PanoramaMoGeExecutor()
    try:
        lvl = int(executor.args_dict.get('logger_level'))
    except (TypeError, ValueError):
        lvl = 20
    executor.logger.setLevel(lvl)
    executor.run()