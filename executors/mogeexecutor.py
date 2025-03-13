import os
import sys

import numpy as np

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

from geoanything.executors.external.mogewraper import MoGeInference


class MoGeExecutor(BaseExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_parser(self):
        super().setup_parser()
        # Optionally, you could add argument definitions here if you wanted
        # command-line usage. But in your environment, they're loaded from base64.

    def run(self):
        self.logger.info(f"Running {self.__class__.__name__}")

        # Model path from args
        model_path = f'{self.args_dict.get("cache_dir")}/model.pt'
        if not model_path or not os.path.exists(model_path):
            self.logger.error(f"Model path does not exist or is not set: {model_path}")
            return

        # The user-specified input pattern
        input_filename_pattern = self.args_dict['inputs']['Input1']

        # ---- NEW: Retrieve optional geometry/camera arguments ----
        sensor_width_mm = float(self.args_dict.get('sensor_width_mm', 36.0))
        sensor_height_mm = float(self.args_dict.get('sensor_height_mm', 24.0))
        user_fov_x = self.args_dict.get('fov_x', None)  # could be None or numeric
        user_focal_length_mm = self.args_dict.get('focal_length_mm', None)  # same

        # Construct MoGeInference with new sensor sizes
        moge_inference = MoGeInference(
            model_path=model_path,
            root_folder='.',  # Or wherever you'd like. Could also use output_dir below.
            device='cuda',
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm
        )

        # The user-specified output pattern
        output_filename_pattern = self.args_dict['output_geo']
        # Prepare frame range
        frame_range = range(self.frame_range[0], self.frame_range[-1] + 1)

        # Create output directory if needed
        output_dir = os.path.dirname(output_filename_pattern)
        os.makedirs(output_dir, exist_ok=True)

        # Read input images (frames) as arrays
        points_list = self.imageIO.read_image(
            input_filename_pattern,
            frame_range=self.frame_range,
            output_format='np'
        )

        # Other existing arguments
        resolution_level = self.args_dict.get('resolution_level', 9)
        apply_mask = self.args_dict.get('apply_mask', False)
        remove_edge = self.args_dict.get('remove_edge', True)
        rtol = float(self.args_dict.get('rtol', 0.02))

        focal_lengths = []

        # Loop over frames
        for img, frame_number in self.logger.progress(zip(points_list, frame_range), desc="Processing frames:"):
            # ---- Pass fov_x/focal_length_mm to the inference call ----
            output = moge_inference.infer(
                img,
                resolution_level=resolution_level,
                apply_mask=apply_mask,
                fov_x=user_fov_x if user_fov_x else None,
                focal_length_mm=float(user_focal_length_mm) if user_focal_length_mm else None
            )

            # Now retrieve the camera data
            camera_data = output.get_camera_data()
            self.logger.info(f'Camera Data: {camera_data}')
            focal_lengths.append(camera_data['fov_x'])

            # Save as OBJ with edges removed if requested
            output_filename = output_filename_pattern % frame_number
            if not output_filename.endswith('.obj'):
                output_filename += '.obj'
            output.save_mesh(output_filename, file_format='obj', remove_edge=remove_edge, rtol=rtol)

        # Return the average fov_x as "focal_length"
        focal_length = np.mean(focal_lengths)
        self.send_output({'output_geo': output_filename_pattern, 'focal_length': focal_length})


if __name__ == '__main__':
    executor = MoGeExecutor()
    try:
        lvl = int(executor.args_dict.get('logger_level'))
    except (TypeError, ValueError):
        lvl = 20
    executor.logger.setLevel(lvl)
    executor.run()
