import cv2
import numpy as np
from mesh import Mesh
from model import Model
from model_registration import ModelRegistration
from pnp_problem import PnPProblem
from robustmatcher import RobustMatcher
import utils
import main_utils

end_registration = False

f = 55                          # focal length in mm
sx = 22.3                       # sensor size           
sy = 14.9                       # sensor size
width = 718                     # image size
height = 480                    # image size

cam_params = {
    "fx": width * f / sx,
    "fy": height * f / sy,
    "cx": width / 2,
    "cy": height / 2
}

n = 7
pts = [1, 2, 3, 5, 6, 7, 8]
registration = ModelRegistration()
model = Model()
mesh = Mesh()
pnp_registration = PnPProblem(cam_params)

def onMouseModelRegistration(event, x, y, flags, param):
    global end_registration
    if event == cv2.EVENT_LBUTTONUP:
        is_registrable = registration.isRegistrable()
        if is_registrable:
            n_regist = registration.getNumRegist()
            n_vertex = pts[n_regist]
            point_2d = np.array([x, y]).astype(np.float32)
            point_3d = mesh.list_vertex[n_vertex - 1]
            registration.registerPoint(point_2d, point_3d)
            if registration.getNumRegist() == registration.getNumMax():
                end_registration = True

img_path = "../Data/mine.jpg"
ply_read_path = "../Data/box.ply"
write_path = "../Data/cookies_ORB2.yml"
numKeyPoints = 9000
featureName = "ORB"

mesh.load(ply_read_path)
useFLANN = False
detector, descriptor = utils.createFeatures(featureName, numKeyPoints)
rmatcher = RobustMatcher(detector, utils.createMatcher(featureName, useFLANN))

cv2.namedWindow("MODEL REGISTRATION", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("MODEL REGISTRATION", onMouseModelRegistration, 0)

img_in = cv2.imread(img_path, cv2.IMREAD_COLOR)
 
if img_in is None:
    print("No img found")
    exit(-1)

num_registrations = n
registration.setNumMax(num_registrations)

red = (0, 0, 255, 0)
green = (0, 255, 0, 0)
blue = (255, 0, 0, 0)
yellow = (0, 255, 255, 0)

while cv2.waitKey(30) < 0:
    img_vis = img_in.copy()

    list_points2d = registration.get_points2d()
    list_points3d = registration.get_points3d()

    utils.drawPoints(img_vis, list_points2d, list_points3d, red)

    if not end_registration:
        n_regist = registration.getNumRegist()
        n_vertex = pts[n_regist]
        current_point3d = mesh.list_vertex[n_vertex - 1]
        utils.drawQuestion(img_vis, current_point3d, green)
        utils.drawCounter(img_vis, registration.getNumRegist(), registration.getNumMax(), red)
    else:
        utils.drawText(img_vis, "END REGISTRATION", green)
        utils.drawCounter(img_vis, registration.getNumRegist(), registration.getNumMax(), red)

    cv2.imshow("MODEL REGISTRATION", img_vis)

print("COMPUTING POSE ...")

list_points2d = registration.get_points2d()
list_points3d = registration.get_points3d()

is_correspondence = pnp_registration.estimatePose(list_points3d, list_points2d, cv2.SOLVEPNP_ITERATIVE)
if is_correspondence:
    print("Correspondence found")
    list_points2d_mesh = pnp_registration.verifyPoints(mesh)
    utils.draw2DPoints(img_vis, list_points2d_mesh, green)
else:
    print("Correspondence not found")

cv2.imshow("MODEL REGISTRATION", img_vis)

cv2.waitKey(0)

keypoints_model, descriptors = rmatcher.computeKeypointsAndDescriptors(img_in)

for i, keypoint in enumerate(keypoints_model):
    point2d = keypoint.pt
    on_surface, point3d = pnp_registration.backproject2DPoint(mesh, point2d)
    if on_surface:
        model.add_correspondence(point2d, point3d)
        model.add_descriptor(descriptors[i])
        model.add_keypoint(keypoint)
    else:
        model.add_outlier(point2d)

model.set_training_image_path(img_path)
model.save(write_path)

img_vis = img_in.copy()

list_points_in = model.get_list_points2d_in()
list_points_out = model.get_list_points2d_out()

num = str(len(list_points_in))
text = "There are {} inliers".format(num)
utils.drawText(img_vis, text, red)

num = str(len(list_points_out))
text = "There are {} outliers".format(num)
utils.drawText2(img_vis, text, red)

utils.drawObjectMesh(img_vis, mesh, pnp_registration, blue)

utils.draw2DPoints(img_vis, list_points_in, green)
utils.draw2DPoints(img_vis, list_points_out, red)

cv2.imshow("MODEL REGISTRATION", img_vis)

cv2.waitKey(0)

cv2.destroyWindow("MODEL REGISTRATION")

print("GOODBYE")


