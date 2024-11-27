import os.path
import sys
import numpy as np

paths = [r'E:\stable-diffusion\stable-diffusion-integrator', r'E:\track_anything_project',
         r'E:\ai_projects\dust3r_project\vision-forge', r'E:\nuke-bridge', 'E:/ai_projects/ai_portal']
for p in paths:
    if not p in sys.path:
        sys.path.append(p)

from nukebridge.executors.core.baseexecutor import BaseExecutor

from geoanything.executors.external.mogewraper import MoGeInference


class MoGeExecutor(BaseExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_parser(self):
        super().setup_parser()
        # self.parser.add_argument('--output', required=True, help='Output path for the generated geometry')

    def run(self):
        self.logger.info(f"Running {self.__class__.__name__}")
        model_path = f'{self.args_dict.get("cache_dir")}/model.pt'
        if model_path is None:
            self.logger.error('Model path is required')
            return
        if not os.path.exists(model_path):
            self.logger.error(f'Model path does not exist: {model_path}')
            return

        input_filename_pattern = self.args_dict['inputs']['Input1']
        moge_inference = MoGeInference(model_path=model_path)
        output_filename_pattern = self.args_dict['output_geo']
        frame_range = range(self.frame_range[0], self.frame_range[-1] + 1)

        output_dir = os.path.dirname(output_filename_pattern)
        os.makedirs(output_dir, exist_ok=True)
        points_list = self.imageIO.read_image(input_filename_pattern, frame_range=self.frame_range, output_format='np')
        resolution_level = self.args_dict.get('resolution_level', 9)
        apply_mask = self.args_dict.get('apply_mask', False)
        remove_edge = self.args_dict.get('remove_edge', True)
        rtol = self.args_dict.get('rtol', 0.02)
        focal_lengths = []
        for img, frame_number in self.logger.progress(zip(points_list, frame_range), desc="Processing frames:"):
            output = moge_inference.infer(img, resolution_level, apply_mask)
            camera_data = output.get_camera_data()

            self.logger.info(f'Camera Data: {camera_data}')
            focal_lengths.append(camera_data['fov_x'])
            output_filename = output_filename_pattern % frame_number
            output_filename = output_filename if output_filename.endswith('.obj') else output_filename + '.obj'
            output.save_mesh(output_filename, file_format='obj', remove_edge=remove_edge, rtol=rtol)
        focal_length = np.mean(focal_lengths)
        self.send_output({'output_geo': output_filename_pattern, 'focal_length':focal_length})


if __name__ == '__main__':
    # Create a BaseExecutor object.
    executor = MoGeExecutor()
    try:
        # Try to get the logger level from the arguments.
        lvl = int(executor.args_dict.get('logger_level'))
    except TypeError:
        # If the logger level is not specified in the arguments, set it to 20.
        lvl = 20
    # Set the logger level.
    executor.logger.setLevel(lvl)
    executor.run()
