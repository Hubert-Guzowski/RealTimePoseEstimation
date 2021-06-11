from typing import List

import cv2
import numpy as np


class Model:
    def __init__(self):
        self.__n_correspondences = 0
        self.__keypoints = []
        self.__list_points2d_in = []
        self.__list_points2d_out = []
        self.__list_points3d_in = []
        self.__descriptors = np.ndarray
        self.__training_img_path = "Data/resized_IMG_3875.jpeg"

    def add_correspondence(self, point2d: np.ndarray, point3d: np.ndarray) -> None:
        self.__list_points2d_in.append(point2d)
        self.__list_points3d_in.append(point3d)
        self.__n_correspondences += 1

    def add_outlier(self, point2d: np.ndarray) -> None:
        self.__list_points2d_in.append(point2d)

    def add_descriptor(self, descriptor: np.ndarray) -> None:
        self.__descriptors = np.vstack(self.__descriptors, descriptor)

    def add_keypoint(self, keypoint: List[float]) -> None:
        self.__keypoints.append(keypoint)

    def get_training_image_path(self):
        return self.__training_img_path

    def get_list_points3d_in(self):
        return self.__list_points3d_in

    def get_descriptors(self):
        return self.__descriptors

    def get_keypoints(self):
        return self.__keypoints

    def set_training_image_path(self, path: str) -> None:
        self.__training_img_path = path

    def save(self, path: str) -> None:
        points_3d_matrix = np.array(self.__list_points3d_in)
        points_2d_matrix = np.array(self.__list_points2d_in)

        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        storage.write("points_3d", points_3d_matrix)
        storage.write("points_2d", points_2d_matrix)
        storage.write("keypoints", self.__keypoints)
        storage.write("descriptors", self.__descriptors)
        storage.write("training_image_path", self.__training_img_path)

        storage.release()

    def load(self, path: str) -> None:
        # Not sure, if that's correct - documentation is sparse
        # Based on: https://www.programcreek.com/python/example/110693/cv2.FileStorage
        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        points3d = storage.getNode("points_3d")
        descriptors = storage.getNode("descriptors")
        self.__list_points3d_in = np.array(points3d.mat().tolist())
        self.__descriptors = np.array(descriptors.mat().tolist())

        def createKeypoint(a, i):
            return cv2.KeyPoint(a[i], a[i+1], a[i+2], a[i+3], a[i+4], int(a[i+5]), int(a[i+6]))


        keypoints = storage.getNode("keypoints")
        if not keypoints.empty():
            loaded = [keypoints.at(i).real() for i in range(keypoints.size())]
            parsed = [createKeypoint(loaded, i) for i in range(0, len(loaded), 7)]
            self.__keypoints = parsed

        training_image_path = storage.getNode("training_image_path")
        if not training_image_path.empty():
            self.__training_img_path = training_image_path.string()

        storage.release()
