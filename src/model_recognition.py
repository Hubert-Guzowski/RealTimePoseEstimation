import numpy as np


class ModelRecognition():
    def __init__(self, n_registrations=0, max_registrations=0):
        self.n_registrations = n_registrations
        self.max_registrations = max_registrations
        self.list_points2d_ = []
        self.list_points3d_ = []

    def registerPoint(self, point2d: np.array, point3d: np.array):
        self.list_points2d_.append(point2d)
        self.list_points3d_.append(point3d)
        self.n_registrations += 1

    def reset(self):
        self.n_registrations = 0
        self.max_registrations = 0
        self.list_points2d_ = []
        self.list_points3d_ = []