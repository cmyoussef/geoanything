import os

import cv2
import numpy as np
import torch
from nukebridge.executors.core.baseexecutor import BaseExecutor

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

    @staticmethod
    def enhance_depth_with_guided_filter(depth, rgb_image, radius=4, eps=1e-6):
        """
        Enhance the depth map using guided filtering with the RGB image as guidance.

        Parameters:
            depth (np.array): Normalized depth map (values between 0 and 1).
            rgb_image (np.array): Corresponding RGB image.
            radius (int): Radius of the guided filter.
            eps (float): Regularization parameter to avoid division by zero.

        Returns:
            np.array: Enhanced depth map.
        """
        import cv2
        # Ensure depth is single-channel and float32
        depth = depth.astype(np.float32)
        if len(depth.shape) == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        # Convert RGB image to float32
        rgb_image = rgb_image.astype(np.float32) / 255.0

        # Apply guided filter
        depth_enhanced = cv2.ximgproc.guidedFilter(
            guide=rgb_image,
            src=depth,
            radius=radius,
            eps=eps,
            dDepth=-1  # Output depth (-1 means same as input)
        )

        return depth_enhanced

    @staticmethod
    def preprocess_depth_map(depth_map):
        """
        Apply bilateral filtering to the depth map.

        Parameters:
            depth_map (np.array): The original depth map.

        Returns:
            np.array: The filtered depth map.
        """
        # Convert depth map to 16-bit unsigned integer if necessary
        depth_map_uint16 = (depth_map * 1000).astype(np.uint16)

        # Apply bilateral filter
        filtered_depth_map = cv2.bilateralFilter(depth_map_uint16, d=9, sigmaColor=75, sigmaSpace=75)

        # Convert back to float
        filtered_depth_map = filtered_depth_map.astype(np.float32) / 1000.0

        return filtered_depth_map

    @staticmethod
    def sharpen_depth_with_unsharp_mask(depth, kernel_size=(3, 3), alpha=2.0, beta=-1.0):
        """
        Sharpen the depth map using unsharp masking.

        Parameters:
            depth (np.array): Normalized depth map (values between 0 and 1).
            kernel_size (tuple): Size of the Gaussian kernel.
            alpha (float): Weight of the original image.
            beta (float): Weight of the blurred image (negative to subtract).

        Returns:
            np.array: Sharpened depth map.
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(depth, kernel_size, 0)

        # Sharpen the image
        depth_sharpened = cv2.addWeighted(depth, alpha, blurred, beta, 0)

        # Clip values to [0, 1]
        depth_sharpened = np.clip(depth_sharpened, 0, 1)

        return depth_sharpened

    @staticmethod
    def enhance_depth_with_edge_sharpening(depth, rgb_image, edge_weight=0.2):
        """
        Enhance the depth map by sharpening edges using edge detection.

        Parameters:
            depth (np.array): Normalized depth map (values between 0 and 1).
            rgb_image (np.array): Corresponding RGB image.
            edge_weight (float): Weight applied to the edge map when enhancing the depth map.

        Returns:
            np.array: Enhanced depth map.
        """
        # Convert RGB image to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0

        # Detect edges using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = cv2.magnitude(grad_x, grad_y)
        edge_map = cv2.normalize(edge_map, None, 0, 1, cv2.NORM_MINMAX)

        # Sharpen depth map using the edge map
        depth_enhanced = depth + edge_weight * edge_map
        depth_enhanced = np.clip(depth_enhanced, 0, 1)

        return depth_enhanced

    def run(self):
        """
        Method to run the executor.
        Handles loading the model, processing the images, and saving the results based on provided arguments.
        """
        self.logger.debug(f"Running {self.__class__.__name__} with arguments: {self.dict_to_string(self.args_dict)}")
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.logger.debug(f"Using {DEVICE} device")
        encoder = 'vitl' # self.args_dict.get('encoder', 'vitl')
        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(
            torch.load(f'{self.args_dict.get("cache_dir")}/depth_anything_v2_{encoder}.pth', map_location=DEVICE))
        depth_model = depth_model.to(DEVICE).eval()
        self.logger.debug(f"Model loaded from {self.args_dict.get('cache_dir')}/depth_anything_v2_{encoder}.pth")
        # Prepare input file list based on frame range pattern
        input_filename_pattern = self.args_dict['inputs']['Input1']
        output_filename_pattern = self.args_dict['output']
        frame_range = range(self.frame_range[0], self.frame_range[-1] + 1)  # Unpack the tuple directly into range
        self.logger.debug(f"Processing {len(frame_range)} frames...")
        self.logger.debug(f"Input file pattern: {input_filename_pattern}")
        self.logger.debug(f"Output file pattern: {output_filename_pattern}, frame range: {frame_range}")
        raw_images = self.imageIO.read_image(input_filename_pattern, frame_range=self.frame_range, output_format='np')
        self.logger.debug(f"loading {len(raw_images)} images")
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename_pattern)
        os.makedirs(output_dir, exist_ok=True)
        # Processing each frame
        # print(f"Processing {len(frame_range)} frames...")
        self.logger.info(f"Processing {len(frame_range)} frames...")
        for frame_number, raw_image in self.logger.progress(zip(frame_range, raw_images)):
            input_filename = input_filename_pattern % frame_number
            output_filename = output_filename_pattern % frame_number
            # raw_image.show()
            if not os.path.exists(input_filename):
                self.logger.warning(f"File not found: {input_filename}")
                continue
            input_size = 518 # self.args_dict.get('input_size', 518)
            # raw_image = self.imageIO.read_image(input_filename)# cv2.imread(input_filename)
            depth = depth_model.infer_image(raw_image, input_size)
            # depth = (depth - depth.min()) / (depth.max() - depth.min())
            # self.imageIO.write_image(depth, output_filename)
            # Normalize and prepare depth image
            # Normalize the depth map to range [0, 1]
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

            # Enhance depth map using guided filter
            depth_guided  = self.enhance_depth_with_guided_filter(depth_normalized, raw_image, radius=6, eps=1e-2)
            depth_sharpened = self.sharpen_depth_with_unsharp_mask(depth_normalized , kernel_size=(3, 3), alpha=2.0, beta=-1.0)
            depth_enhanced = self.enhance_depth_with_edge_sharpening(depth_sharpened, raw_image, edge_weight=0.15)

            output_name, ext = os.path.splitext(output_filename)
            self.imageIO.write_image(depth_normalized, f'{output_name}.exr')
            self.imageIO.write_image(depth_guided, f'{output_name}e.exr')
            self.imageIO.write_image(depth_sharpened, f'{output_name}s.exr')
            self.imageIO.write_image(depth_enhanced, f'{output_name}e.exr')
            # depth = (depth_enhanced * 65535.0).astype(np.uint16)

            #
            # cv2.imwrite(output_filename, depth)

            self.logger.info(f'Processed frame {frame_number} saved to {output_filename}')


# It creates a DepthExecutor object and sets the logger level.
if __name__ == '__main__':
    # Create a DepthExecutor object.
    executor = DepthAnythingExecutor()

    # print(executor.args_dict)
    try:
        # Try to get the logger level from the arguments.
        lvl = int(executor.args_dict.get('logger_level', 20))
    except TypeError:
        # If the logger level is not specified in the arguments, set it to 20.
        lvl = 20
    # Set the logger level.
    executor.logger.setLevel(lvl)
    print('runing', '*'*100)
    executor.run()
