import numpy as np

class PnPProblem:
    def backproject3DPoint(self, point3d: np.array):
        pass

    def estimatePoseRANSAC(self,
            list_points3d_model_match,
            list_points2d_scene_match,
            pnpMethod,
            iterationsCount,
            reprojectionError,
            confidence):
        pass