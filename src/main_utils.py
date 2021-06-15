import cv2
import numpy as np
import utils

def initKalmanFilter(nStates, nMeasurements, nInputs, dt):
    KF = cv2.KalmanFilter(nStates, nMeasurements, nInputs)
    KF.processNoiseCov = cv2.setIdentity(KF.processNoiseCov, 1e-5)
    KF.measurementNoiseCov = cv2.setIdentity(KF.measurementNoiseCov, 1e-2)
    KF.errorCovPost = cv2.setIdentity(KF.errorCovPost, 1)
    dt2 = 0.5*dt*dt
    transitionMatrix = np.identity(nStates, dtype=np.float32)
    transitionMatrix[0, 3] = dt
    transitionMatrix[1, 4] = dt
    transitionMatrix[2, 5] = dt
    transitionMatrix[3, 6] = dt
    transitionMatrix[4, 7] = dt
    transitionMatrix[5, 8] = dt
    transitionMatrix[0, 6] = dt2
    transitionMatrix[1, 7] = dt2
    transitionMatrix[2, 8] = dt2

    transitionMatrix[9, 12] = dt
    transitionMatrix[10, 13] = dt
    transitionMatrix[11, 14] = dt
    transitionMatrix[12, 15] = dt
    transitionMatrix[13, 16] = dt
    transitionMatrix[14, 17] = dt
    transitionMatrix[9, 15] = dt2
    transitionMatrix[10, 16] = dt2
    transitionMatrix[11, 17] = dt2

    KF.transitionMatrix = transitionMatrix

    measurementMatrix = np.zeros((nMeasurements, nStates), dtype=np.float32)
    measurementMatrix[0, 0] = 1
    measurementMatrix[1, 1] = 1
    measurementMatrix[2, 2] = 1
    measurementMatrix[3, 9] = 1
    measurementMatrix[4, 10] = 1
    measurementMatrix[5, 11] = 1

    KF.measurementMatrix = measurementMatrix

    return KF

def updateKalmanFilter(KF, measurement):
    translation_estimated = np.zeros(3, dtype=np.float32)
    rotation_estimated = np.zeros((3, 3), dtype=np.float32)

    predicition = KF.predict()
    estimated = KF.correct(measurement)

    translation_estimated[0] = estimated[0]
    translation_estimated[1] = estimated[1]
    translation_estimated[2] = estimated[2]

    eulers_estimated = np.zeros(3, dtype=np.float32)
    eulers_estimated[0] = estimated[9]
    eulers_estimated[1] = estimated[10]
    eulers_estimated[2] = estimated[11]

    rotation_estimated = utils.euler2rot(eulers_estimated)

    return translation_estimated, rotation_estimated

def fillMeasurements(translation_measured, rotation_measured):
    measurements = np.zeros(6, dtype=np.float32)
    measured_eulers = utils.rot2euler(rotation_measured)

    measurements[0] = translation_measured[0]
    measurements[1] = translation_measured[1]
    measurements[2] = translation_measured[2]
    measurements[3] = measured_eulers[0]
    measurements[4] = measured_eulers[1]
    measurements[5] = measured_eulers[2]

    return measurements



