import cv2
import numpy as np
import math
from mesh import Mesh
from pnp_problem import PnPProblem

fontFace = cv2.FONT_ITALIC
fontScale = 0.75
thickness_font = 2

lineType = 8
radius = 4


def drawQuestion(image, point: np.array, color):
    text = "Where is point {}?".format(str(point))
    image = cv2.putText(image, text, (25, 50), fontFace, fontScale, color, thickness_font, 8)


def drawText(image, text, color):
    image = cv2.putText(image, text, (25, 50), fontFace, fontScale, color, thickness_font, 8)


def drawText2(image, text, color):
    image = cv2.putText(image, text, (25, 75), fontFace, fontScale, color, thickness_font, 8)


def drawFPS(image, fps, color):
    fps_str = "{:.2f}".format(fps)
    image = cv2.putText(image, fps_str, (500, 50), fontFace, fontScale, color, thickness_font, 8)


def drawConfidence(image, convidence, color):
    text  = str(convidence) + '%'
    image = cv2.putText(image, text , (500, 75), fontFace, fontScale, color, thickness_font, 8)


def drawCounter(image, n, n_max, color):
    text = str(n) + " of " + str(n_max) + " points"
    image = cv2.putText(image, text , (500, 75), fontFace, fontScale, color, thickness_font, 8)


def drawPoints(image, list_points_2d, list_points_3d, color):
    for i in range(len(list_points_2d)):
        point_2d = list_points_2d[i]
        point_3d = list_points_3d[i]

        idx = str(i + 1)
        x = str(point_3d[0])
        y = str(point_3d[1])
        z = str(point_3d[2])
        text = "P {} ({}, {}, {})".format(idx, x, y, z)

        point_2d[0] = point_2d[0] + 10
        point_2d[1] = point_2d[1] - 10
        image = cv2.putText(image, text, (500, 75), fontFace, fontScale, color, thickness_font, 8)


def draw2DPoints(image, list_points_2d, color):
    for point_2d in list_points_2d:
        image = cv2.circle(image, tuple(point_2d), radius, color, -1, lineType)


def drawArrow(image, p: np.array, q: np.array, color, arrowMagnitude = 9, thickness = 1, line_type = 8, shift = 0):
    image = cv2.line(image, tuple(p), tuple(q), color, thickness, line_type, shift)

    angle = math.atan2(p[1] - q[1], p[0] - q[0])
    p[0] = int(q[0] + arrowMagnitude * math.cos(angle + math.pi/4))
    p[1] = int(q[1] + arrowMagnitude * math.sin(angle + math.pi/4))
    image = cv2.line(image, tuple(p), tuple(q), color, thickness, line_type, shift)

    p[0] = int(q[0] + arrowMagnitude * math.cos(angle - math.pi/4))
    p[1] = int(q[1] + arrowMagnitude * math.sin(angle - math.pi/4))
    image = cv2.line(image, tuple(p), tuple(q), color, thickness, line_type, shift)


def draw3DCoordinateAxes(image, list_points_2d):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    black = (0, 0, 0)

    origin = list_points_2d[0]
    pointX = list_points_2d[1]
    pointY = list_points_2d[2]
    pointZ = list_points_2d[3]

    drawArrow(image, origin, pointX, red, 9, 2)
    drawArrow(image, origin, pointY, green, 9, 2)
    drawArrow(image, origin, pointZ, blue, 9, 2)
    image = cv2.circle(image, origin, radius/2, black, -1, lineType)


def drawObjectMesh(image, mesh: Mesh, pnpProblem: PnPProblem, color):
    list_triangles = mesh.list_triangles
    for i in range(len(list_triangles)):
        tmp_triangle = list_triangles[i]

        point_3d_0 = mesh.list_vertex[tmp_triangle[0]]
        point_3d_1 = mesh.list_vertex[tmp_triangle[1]]
        point_3d_2 = mesh.list_vertex[tmp_triangle[2]]

        point_2d_0 = pnpProblem.backproject3DPoint(point_3d_0)
        point_2d_1 = pnpProblem.backproject3DPoint(point_3d_1)
        point_2d_2 = pnpProblem.backproject3DPoint(point_3d_2)

        image = cv2.line(image, point_2d_0, point_2d_1, color, 1)
        image = cv2.line(image, point_2d_1, point_2d_2, color, 1)
        image = cv2.line(image, point_2d_2, point_2d_0, color, 1)


def get_translation_error(t_true, t):
    return cv2.norm(t_true - t)


def get_rotation_error(R_true: np.matrix, R: np.matrix):
    error_mat = -R_true * R.T
    error_vec, _ = cv2.Rodrigues(error_mat)
    return error_vec


def rot2euler(rotationMatrix: np.matrix):
    m00 = rotationMatrix[0][0]
    m02 = rotationMatrix[0][2]
    m10 = rotationMatrix[1][0]
    m11 = rotationMatrix[1][1]
    m12 = rotationMatrix[1][2]
    m20 = rotationMatrix[2][0]
    m22 = rotationMatrix[2][2]

    if m10 > 0.998:
        bank = .0
        attitude = math.pi / 2
        heading = math.atan2(m02, m22)
    elif m10 < -0.998:
        bank = .0
        attitude = - math.pi / 2
        heading = math.atan2(m02, m22)
    else:
        bank = math.atan2(-m12, m11)
        attitude = math.asin(m10)
        heading = math.atan2(m20, m00)

    euler = np.array([bank, attitude, heading])
    return euler


def euler2rot(euler: np.array):
    bank = euler[0]
    attitude = euler[1]
    heading = euler[2]

    ch = math.cos(heading)
    sh = math.sin(heading)
    ca = math.cos(attitude)
    sa = math.sin(attitude)
    cb = math.cos(bank)
    sb = math.sin(bank)

    m00 = ch * ca
    m01 = sh * sb - ch * sa * cb
    m02 = ch * sa * sb + sh * cb
    m10 = sa
    m11 = ca * cb
    m12 = -ca * sb
    m20 = -sh * ca
    m21 = sh * sa * cb + ch * sb
    m22 = -sh * sa * sb + ch * cb

    rotationMatrix = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
    return rotationMatrix


def createFeatures(featureName, numKeypoints):
    if featureName == "ORB":
        detector = cv2.ORB_create(nfeatures=numKeypoints)
        descriptor = cv2.ORB_create(nfeatures=numKeypoints)
        return detector, descriptor
    elif featureName == "KAZE":
        detector = cv2.KAZE_create()
        descriptor = cv2.KAZE_create()
        return detector, descriptor
    elif featureName == "AKAZE":
        detector = cv2.AKAZE_create()
        descriptor = cv2.AKAZE_create()
        return detector, descriptor
    elif featureName == "BRISK":
        detector = cv2.BRISK_create()
        descriptor = cv2.BRISK_create()
        return detector, descriptor


def createMatcher(featureName, useFLANN):
    if featureName == "ORB" or featureName == "AKAZE" or featureName == "BRISK":
        if useFLANN:
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 1) #2
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    else:
        if useFLANN:
            return cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
        else:
            return cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
