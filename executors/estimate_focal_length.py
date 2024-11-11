import cv2
import numpy as np
from nukebridge.executors.core.baseexecutor import BaseExecutor
from sklearn.cluster import KMeans


class FocalLengthEstimator(BaseExecutor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        self.logger.info(f"Running {self.__class__.__name__} with arguments: {self.dict_to_string(self.args_dict)}")

        image_path = self.args_dict['inputs']['Input1']
        sensor_width_mm = self.args_dict.get('sensor_width_mm', 36.0)
        visualize = self.args_dict.get('visualize', False)

        images = self.imageIO.read_image(image_path, output_format='np', frame_range=self.frame_range)
        # Load the image
        if len(images) > 0:
            img = images[0]  # cv2.imread(image_path)
        else:
            raise ValueError("Image not found or unable to load.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_display = img.copy()  # Copy for visualization if needed

        # Edge detection with optimized thresholds
        edges = cv2.Canny(gray, 100, 200, apertureSize=3)

        # Line detection using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100,
                                minLineLength=50, maxLineGap=5)
        if lines is None:
            raise ValueError("No lines detected in the image.")
        lines = lines[:, 0, :]  # Simplify the array shape

        # Store line angles and points
        angles = []
        line_points = []

        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx)
            angles.append(angle)
            line_points.append(((x1, y1), (x2, y2)))
        angles = np.array(angles)

        # Cluster angles into three groups
        num_clusters = 3
        angles_deg = np.degrees(angles).reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(angles_deg)
        labels = kmeans.labels_

        # Group lines according to clusters
        line_groups = {}
        for i in range(num_clusters):
            line_groups[i] = []

        for idx, label in enumerate(labels):
            line_groups[label].append(line_points[idx])

        # Visualization: Draw lines
        if visualize:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for idx, label in enumerate(labels):
                (x1, y1), (x2, y2) = line_points[idx]
                color = colors[label % len(colors)]
                cv2.line(img_display, (x1, y1), (x2, y2), color, 3)

        # Compute vanishing points
        def compute_vanishing_point(lines):
            intersections = []
            num_lines = len(lines)
            for i in range(num_lines):
                for j in range(i + 1, num_lines):
                    l1_p1, l1_p2 = lines[i]
                    l2_p1, l2_p2 = lines[j]

                    # Compute lines in homogeneous coordinates
                    l1 = np.cross([*l1_p1, 1], [*l1_p2, 1])
                    l2 = np.cross([*l2_p1, 1], [*l2_p2, 1])

                    # Compute intersection (vanishing point)
                    vp = np.cross(l1, l2)
                    if vp[2] != 0:
                        vp = vp / vp[2]
                        intersections.append(vp[:2])  # Only x and y
            if intersections:
                # Use median to reduce the effect of outliers
                intersections = np.array(intersections)
                vp_estimate = np.median(intersections, axis=0)
                return vp_estimate
            else:
                return None

        vanishing_points = {}
        for label in line_groups:
            vp = compute_vanishing_point(line_groups[label])
            if vp is not None:
                vanishing_points[label] = vp

                # Visualization: Draw vanishing points
                if visualize:
                    vp_int = tuple(map(int, vp))
                    cv2.circle(img_display, vp_int, 5, (0, 255, 255), -1)

        # Get image dimensions
        h, w = gray.shape
        cx, cy = w / 2, h / 2

        if len(vanishing_points) < 2:
            raise ValueError("Not enough vanishing points detected.")

        # Prepare combinations of vanishing points
        vp_keys = list(vanishing_points.keys())
        focal_lengths = []

        for i in range(len(vp_keys)):
            for j in range(i + 1, len(vp_keys)):
                vp1 = vanishing_points[vp_keys[i]]
                vp2 = vanishing_points[vp_keys[j]]

                # Compute focal length using corrected formula
                dx1 = vp1[0] - cx
                dy1 = vp1[1] - cy
                dx2 = vp2[0] - cx
                dy2 = vp2[1] - cy

                numerator = - (dx1 * dx2 + dy1 * dy2)
                f_squared = numerator
                if f_squared > 0:
                    focal_lengths.append(np.sqrt(f_squared))

        if focal_lengths:
            # Take the median focal length as the estimate
            focal_length_pixels = np.median(focal_lengths)
        else:
            raise ValueError("Failed to compute a valid focal length.")

        # Convert focal length from pixels to millimeters
        sensor_width_mm = float(sensor_width_mm)
        focal_length_mm = focal_length_pixels * (sensor_width_mm / w)

        # Visualization: Show image with lines and vanishing points
        if visualize:
            cv2.imshow('Lines and Vanishing Points', img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.send_output({'focal_length_mm': focal_length_mm})
        return focal_length_mm


if __name__ == '__main__':
    # Create a BaseExecutor object.
    executor = FocalLengthEstimator()
    try:
        # Try to get the logger level from the arguments.
        lvl = int(executor.args_dict.get('logger_level'))
    except TypeError:
        # If the logger level is not specified in the arguments, set it to 20.
        lvl = 20
    # Set the logger level.
    executor.logger.setLevel(lvl)
    executor.run()
