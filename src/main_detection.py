import cv2
import numpy as np
from mesh import Mesh
from model import Model
from pnp_problem import PnPProblem
from robustmatcher import RobustMatcher
import utils
import main_utils

# Settings

video_read_path = "../Data/box.mp4"
yml_read_path = "../Data/cookies_ORB.yml"
ply_read_path = "../Data/box.ply"

f = 55                          # focal length in mm
sx = 22.3                       # sensor size           
sy = 14.9                       # sensor size
width = 640                     # image size
height = 480                    # image size

cam_params = {
    "fx": width * f / sx,
    "fy": height * f / sy,
    "cx": width / 2,
    "cy": height / 2
}

red = (0, 0, 255, 0)
green = (0, 255, 0, 0)
blue = (255, 0, 0, 0)
yellow = (0, 255, 255, 0)


# Robust Matcher parameters
numKeyPoints = 2000
ratioTest = .70
fast_match = True

# RANSAC parameters
iterationsCount = 500
reprojectionError = 6.0
confidence = 0.99

# Kalman Filter parameters
minInliersKalman = 30

# PnP parameters
pnpMethod = None # TODO: Rewrite PnP
featureName = "ORB"
useFLANN = False

# Save results
saveDirectory = ""
frameSave = None # TODO
frameCount = 0

displayFilteredPose = False

# Initialization

pnp_detection = PnPProblem(cam_params)
pnp_detection_est = None # TODO

model = Model()
model.load(yml_read_path)

mesh = Mesh()
mesh.load(ply_read_path)

detector, descriptor = utils.createFeatures(featureName, numKeyPoints)
rmatcher = RobustMatcher(detector, utils.createMatcher(featureName, useFLANN), ratioTest, cv2.imread(model.get_training_image_path()))

nStates = 18
nMeasurements = 6
nInputs = 0
dt = 0.125

KF = main_utils.initKalmanFilter(nStates, nMeasurements, nInputs, dt)

good_measurement = False

list_points3d_model = model.get_list_points3d_in()
descriptors_model = model.get_descriptors()
keypoints_model = model.get_keypoints()

cv2.namedWindow("REAL TIME DEMO", cv2.WINDOW_KEEPRATIO)

cap = cv2.VideoCapture()
cap.open(video_read_path)

# TODO: 186 - 199 Saving video

tm = cv2.TickMeter()

while cv2.waitKey(30) != 27:
    ret, frame = cap.read()
    if ret == 0:
        break
    tm.reset()
    tm.start()
    frame_vis = np.copy(frame)

    if fast_match is True:
        good_matches, keypoints_scene = rmatcher.fastRobustMatch(frame, descriptors_model, keypoints_model)
    else:
        good_matches, keypoints_scene = rmatcher.robustMatch(frame, descriptors_model, keypoints_model)

    frame_matching = rmatcher.getImageMatching()

    if frame_matching is not None:
        cv2.imshow("Keypoints matching", frame_matching)

    list_points3d_model_match = []
    list_points2d_scene_match = []

    for match in good_matches:
        point3d_model = list_points3d_model[match.trainIdx]
        point2d_scene = keypoints_scene[match.queryIdx].pt
        list_points3d_model_match.append(point3d_model)
        list_points2d_scene_match.append(point2d_scene)


    utils.draw2DPoints(frame_vis, list_points2d_scene_match, red) # TODO

    good_measurement = False
    list_points2d_inliers = []

    if len(good_matches) >= 4:
        inliers_idx = pnp_detection.estimatePoseRANSAC( 
            np.array(list_points3d_model_match),
            np.array(list_points2d_scene_match),
            pnpMethod,
            iterationsCount, 
            reprojectionError, 
            confidence)

        for n in inliers_idx:
            point2d = list_points2d_scene_match[n[0]]
            list_points2d_inliers.append(point2d)

        utils.draw2DPoints(frame_vis, list_points2d_inliers, blue)

        if len(inliers_idx) >= minInliersKalman:
            translation_measured = pnp_detection.get_t_matrix()
            rotation_measured = pnp_detection.get_R_matrix()
            measurements = main_utils.fillMeasurements(translation_measured, rotation_measured)
            good_measurement = True

        translation_estimated, rotation_estimated = main_utils.updateKalmanFilter(KF, measurements)
        pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated)

    l = 5
    pose_points2d = []
    if not good_measurement or displayFilteredPose:
        utils.drawObjectMesh(frame_vis, mesh, pnp_detection_est, yellow)
        pose_points2d.append(pnp_detection_est.backproject3DPoint((0, 0, 0)))
        pose_points2d.append(pnp_detection_est.backproject3DPoint((l, 0, 0)))
        pose_points2d.append(pnp_detection_est.backproject3DPoint((0, l, 0)))
        pose_points2d.append(pnp_detection_est.backproject3DPoint((0, 0, l)))
        utils.draw3DCoordinateAxes(frame_vis, pose_points2d)
    else:
        utils.drawObjectMesh(frame_vis, mesh, pnp_detection, green)
        pose_points2d.append(pnp_detection.backproject3DPoint((0, 0, 0)))
        pose_points2d.append(pnp_detection.backproject3DPoint((l, 0, 0)))
        pose_points2d.append(pnp_detection.backproject3DPoint((0, l, 0)))
        pose_points2d.append(pnp_detection.backproject3DPoint((0, 0, l)))
        utils.draw3DCoordinateAxes(frame_vis, pose_points2d)

    tm.stop()

    fps = 1 / tm.getTimeSec()
    utils.drawFPS(frame_vis, fps, yellow)

    detection_ratio = (inliers_idx.size() / good_matches.size()) * 100
    utils.drawConfidence(frame_vis, detection_ratio, yellow)

    inliers_int = inliers_idx.size()
    outliers_int = good_matches.size() - inliers_idx.size()

    text = "Found {} of {} matches".format(inliers_int, good_matches.size())
    text2 = "Inliers: {} - Outliers: {}".format(inliers_int, outliers_int)
    utils.drawText(frame_vis, text, green)
    utils.drawText2(frame_vis, text2, red)

    cv2.imshow("REAL TIME DEMO", frame_vis)

    # TODO - 345:367 Saving video

cv2.destroyWindow("REAL TIME DEMO")
print("GOODBYE ...")


        




        



    









