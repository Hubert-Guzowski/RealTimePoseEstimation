import csv
from typing import List

import numpy as np


class CsvWriter:
    def __init__(self, path, separator=' '):
        self.path = path
        self.separator = separator
        self.is_first_term = True

    def write_xyz(self, list_points3d: List[np.ndarray]) -> None:
        with open(self.path, 'w', newline='') as file:
            writer = csv.writer(file)
            for point in list_points3d:
                writer.writerow(point.tolist())

    def write_uvxyz(self, list_points3d: List[np.ndarray], list_points2d: List[np.ndarray], descriptors: np.ndarray) -> None:
        """ Original version had cv::Mat, which has no direct equivalent in cv2.
        I used numpy array according to: https://answers.opencv.org/question/87233/what-is-the-equivalent-of-cvmat-in-python/"""

        with open(self.path, 'w', newline='') as file:
            writer = csv.writer(file)
            for index in range(len(list_points3d)):
                point3d = list_points3d[index]
                point2d = list_points2d[index]
                row = [point2d[0], point2d[1], point3d[0], point3d[1], point3d[2]]
                row.extend(descriptors[index, 0:32].tolist())
                writer.writerow(row)


# writer = CsvWriter("test")
# writer.write_uvxyz([np.array([0, 1, 2])], [np.array([0, 1])], np.zeros((2, 60)))
