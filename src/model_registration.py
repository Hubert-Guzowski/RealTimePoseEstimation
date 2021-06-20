import numpy as np


class ModelRegistration():
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

    def isRegistrable(self):
        return self.n_registrations < self.max_registrations

    def getNumRegist(self):
        return self.n_registrations

    def getNumMax(self):
        return self.max_registrations

    def get_points2d(self):
        return np.array(self.list_points2d_)
    
    def get_points3d(self):
        return np.array(self.list_points3d_)

    def setNumMax(self, n):
        self.max_registrations = n