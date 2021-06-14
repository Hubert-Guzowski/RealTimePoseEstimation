from typing import Tuple, List

import cv2
import numpy as np

from mesh import Mesh, Ray, Triangle


def get_nearest_3D_point(points_list: List[np.array], origin: np.array):
    p1 = points_list[0]
    p2 = points_list[1]

    d1 = np.linalg.norm(p1 - origin)
    d2 = np.linalg.norm(p2 - origin)

    if d1 < d2:
        return p1
    else:
        return p2


class PnPProblem:

    def __init__(self, camera_params):
        self.A_matrix_ = np.zeros((3, 3))
        self.A_matrix_[0, 0] = camera_params["fx"]
        self.A_matrix_[1, 1] = camera_params["fy"]
        self.A_matrix_[0, 2] = camera_params["cx"]
        self.A_matrix_[1, 2] = camera_params["cy"]
        self.A_matrix_[2, 2] = 1
        self.R_matrix_ = np.zeros((3, 3))
        self.t_matrix_ = np.zeros((3, 1))
        self.P_matrix_ = np.zeros((3, 4))

    def get_A_matrix(self):
        return self.A_matrix_

    def get_R_matrix(self):
        return self.R_matrix_

    def get_t_matrix(self):
        return self.t_matrix_

    def get_P_matrix(self):
        return self.P_matrix_

    def set_P_matrix(self, R_matrix_, t_matrix_):
        self.P_matrix_[0, 0] = R_matrix_[0, 0]
        self.P_matrix_[0, 1] = R_matrix_[0, 1]
        self.P_matrix_[0, 2] = R_matrix_[0, 2]
        self.P_matrix_[1, 0] = R_matrix_[1, 0]
        self.P_matrix_[1, 1] = R_matrix_[1, 1]
        self.P_matrix_[1, 2] = R_matrix_[1, 2]
        self.P_matrix_[2, 0] = R_matrix_[2, 0]
        self.P_matrix_[2, 1] = R_matrix_[2, 1]
        self.P_matrix_[2, 2] = R_matrix_[2, 2]
        self.P_matrix_[0, 3] = t_matrix_[0]
        self.P_matrix_[1, 3] = t_matrix_[1]
        self.P_matrix_[2, 3] = t_matrix_[2]

    def estimatePose(self,
                     list_points3d_model_match,
                     list_points2d_scene_match,
                     pnpMethod):

        distCoeffs = np.zeros((4, 1))
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))

        correspondence = cv2.solvePnP(list_points3d_model_match, list_points2d_scene_match, self.A_matrix_, distCoeffs,
                                      rvec=rvec, tvec=tvec, useExtrinsicGuess=False, flags=pnpMethod)

        cv2.Rodrigues(rvec, self.R_matrix_)

        self.t_matrix_ = tvec
        self.set_P_matrix(self.R_matrix_, self.t_matrix_)

        return correspondence

    def estimatePoseRANSAC(self,
                           list_points3d_model_match,
                           list_points2d_scene_match,
                           pnpMethod,
                           iterationsCount,
                           reprojectionError,
                           confidence):

        distCoeffs = np.zeros((4, 1))
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))

        retval, cameraMatrix, rvec, tvec \
            = cv2.solvePnPRansac(list_points3d_model_match, list_points2d_scene_match, self.A_matrix_, distCoeffs,
                                 rvec=rvec, tvec=tvec, useExtrinsicGuess=False, iterationsCount=iterationsCount,
                                 reprojectionError=reprojectionError, confidence=confidence, flags=pnpMethod)

        cv2.Rodrigues(rvec, self.R_matrix_)

        self.t_matrix_ = tvec
        self.set_P_matrix(self.R_matrix_, self.t_matrix_)

    def backproject3DPoint(self, point3d: np.array):
        point_3d = np.ones((4, 1))
        point_3d[:point3d.shape[0]] = point_3d

        point_2d = self.A_matrix_ * self.P_matrix_ * point_3d

        normalized_point2d = np.zeros((2, 1))
        normalized_point2d[0] = point_2d[0] / point_2d[2]
        normalized_point2d[1] = point_2d[1] / point_2d[2]

        return normalized_point2d

    def backproject2DPoint(self, mesh: Mesh, point2d: np.array, point3d: np.array) -> Tuple[bool, np.array]:
        triangles_list = mesh.list_triangles

        lmb = 8
        u = point2d[0]
        v = point2d[1]

        point2d_vec = np.ones((3, 1))
        point2d_vec[0] = u * lmb
        point2d_vec[1] = v * lmb
        point2d_vec[2] = lmb

        X_c = np.linalg.inv(self.A_matrix_) * point2d_vec
        X_w = np.linalg.inv(self.R_matrix_) * (X_c - self.t_matrix_)
        C_op = np.multiply(np.linalg.inv(self.R_matrix_), -1) * self.t_matrix_

        ray = X_w - C_op
        ray = ray / np.linalg.norm(ray)

        R = Ray(C_op, ray)

        intersections_list = []

        for i in range(len(triangles_list)):
            V0 = mesh.list_vertex[triangles_list[i][0]]
            V1 = mesh.list_vertex[triangles_list[i][1]]
            V2 = mesh.list_vertex[triangles_list[i][2]]

            T = Triangle(V0, V1, V2)

            flag, out = self.intersect_MollerTrumbore(R, T)

            if flag:
                tmp_pt = R.P0 + out * R.P1
                intersections_list.append(tmp_pt)

        if len(intersections_list) > 0:
            point3d = get_nearest_3D_point(intersections_list, R.P0)
            return True, point3d
        else:
            return False, np.empty((1, 1))

    def intersect_MollerTrumbore(self, ray: Ray, triangle: Triangle) -> Tuple[bool, float]:
        EPSILON = 0.000001

        V1 = triangle.V0
        V2 = triangle.V1
        V3 = triangle.V2

        O = ray.P0
        D = ray.P1

        e1 = np.subtract(V2, V1)
        e2 = np.subtract(V3, V1)

        P = np.cross(D, e2)
        det = np.matmul(e1, P)

        if det > -EPSILON and det > EPSILON:
            return False, 0.0
        inv_det = 1.0 / det

        T = np.subtract(O, V1)
        u = np.matmul(T, P) * inv_det

        if u < 0.0 or u > 1.0:
            return False, 0.0

        Q = np.cross(T, e1)
        v = np.matmul(D, Q) * inv_det

        if v < 0.0 or u + v > 1.0:
            return False, 0.0

        t = np.matmul(e2, Q) * inv_det

        if t > EPSILON:
            return True, t

        return False, 0.0
