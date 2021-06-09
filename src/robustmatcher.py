import cv2
import numpy as np

class RobustMatcher:  # TODO
    def __init__(self, featureDetector, descriptorExtractor, descriptorMatcher, ratio, trainingImage):
        self.featureDetector = featureDetector
        self.descriptorExtractor = descriptorExtractor
        self.descriptorMatcher = descriptorMatcher
        self.ratio = ratio
        self.trainingImage = trainingImage

    def computeKeyPoints(self, image, keypoints):
        pass

    def computeDescriptors(self, image, keypoints, descriptors):
        pass

    def getImageMatching(self):
        pass

    def ratioTest(self, matches):
        pass

    def symmetryTest(self, matches1, matches2, symMatches):
        pass

    def robustMatch(self, frame, descriptorsModel, keypointsModel):
        pass

    def fastRobustMatch(self, frame, descriptorsModel, keypointsModel):
        pass


    