import os
import sys


import numpy as np
import cv2
import torch
from nukebridge.executors.core.baseexecutor import BaseExecutor, dict_to_string

# add the path to the sys
__package__ = BaseExecutor.get_relative_pacakge(__file__, index=2)

from .external.depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingExecutor(BaseExecutor):
    """
    DepthExecutor is a class that inherits from the BaseExecutor class.
    It is used to execute depth estimation tasks using the DepthAnythingV2 model.
    """

    def __init__(self):
        """
        Constructor method for the DepthExecutor class.
        Calls the constructor of the BaseExecutor class.
        """
        super().__init__()

    def setup_parser(self):
        """
        Method to set up the parser.
        Extends the setup_parser method of the BaseExecutor class to include specific arguments for depth estimation.
        """
        super().setup_parser()
        self.parser.add_argument('--input-size', type=int, default=518, help='Input size for image resizing')
        self.parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                                 help='Model encoder type')
        self.parser.add_argument('--grayscale', action='store_true', help='Output depth map in grayscale')

    def run(self):
        """
        Method to run the executor.
        Handles loading the model, processing the images, and saving the results based on provided arguments.
        """
        self.logger.debug(f"Running {self.__class__.__name__} with arguments: {dict_to_string(self.args_dict)}")
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        depth_model = DepthAnythingV2(**model_configs[self.args_dict['encoder']])
        depth_model.load_state_dict(
            torch.load(f'{self.args_dict.get("cache_dir")}/depth_anything_v2_{self.args_dict["encoder"]}.pth',
                       map_location=DEVICE))
        depth_model = depth_model.to(DEVICE).eval()

        # Prepare input file list based on frame range pattern
        input_filename_pattern = self.args_dict['inputs']['Input1']
        output_filename_pattern = self.args_dict['output']
        frame_range = range(self.frame_range[0], self.frame_range[-1]+1)  # Unpack the tuple directly into range
        raw_images = self.imageIO.read_image(input_filename_pattern, frame_range=self.frame_range, output_format='cv2')

        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename_pattern)
        os.makedirs(output_dir, exist_ok=True)
        # Processing each frame
        for frame_number, raw_image in self.logger.progress(zip(frame_range, raw_images)):
            input_filename = input_filename_pattern % frame_number
            output_filename = output_filename_pattern % frame_number
            # raw_image.show()
            if not os.path.exists(input_filename):
                self.logger.warning(f"File not found: {input_filename}")
                continue

            # raw_image = self.imageIO.read_image(input_filename)# cv2.imread(input_filename)
            depth = depth_model.infer_image(raw_image, self.args_dict['input_size'])
            # depth = (depth - depth.min()) / (depth.max() - depth.min())
            # self.imageIO.write_image(depth, output_filename)
            # Normalize and prepare depth image
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535.0  # Adjust for 16-bit range
            depth = depth.astype(np.uint16)
            #
            cv2.imwrite(output_filename, depth)

            self.logger.info(f'Processed frame {frame_number} saved to {output_filename}')


# It creates a DepthExecutor object and sets the logger level.
if __name__ == '__main__':
    # Create a DepthExecutor object.
    executor = DepthAnythingExecutor()
    try:
        # Try to get the logger level from the arguments.
        lvl = int(executor.args_dict.get('logger_level', 20))
    except TypeError:
        # If the logger level is not specified in the arguments, set it to 20.
        lvl = 20
    # Set the logger level.
    executor.logger.setLevel(lvl)
    executor.run()
