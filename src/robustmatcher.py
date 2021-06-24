import cv2
import numpy as np

class RobustMatcher: 
    def __init__(self, featureDetector, descriptorMatcher, ratio = .80, trainingImage=None):
        self.featureDetector = featureDetector
        self.descriptorMatcher = descriptorMatcher
        self.ratio = ratio
        self.trainingImage = trainingImage
        self.__imageMatching = None

    def computeKeypointsAndDescriptors(self, image):
        return self.featureDetector.detectAndCompute(image, None)

    def ratioTest(self, matches):
        return [(x,y) for x, y in matches if x.distance / y.distance <= self.ratio]

    def symmetryTest(self, matches1, matches2):
        return [x for (x, y) in matches1 for (a, b) in matches2 if x.queryIdx == a.queryIdx and x.trainIdx == a.trainIdx]

    def getImageMatching(self):
        return self.__imageMatching

    def robustMatch(self, frame, descriptorsModel, keypointsModel):
        keypoints_frame, descriptors_frame = self.computeKeypointsAndDescriptors(frame)

        matches1 = self.descriptorMatcher.knnMatch(descriptors_frame.astype(np.uint8), descriptorsModel.astype(np.uint8), 2)
        matches1 = self.ratioTest(matches1)

        matches2 = self.descriptorMatcher.knnMatch(descriptorsModel.astype(np.uint8), descriptors_frame.astype(np.uint8), 2)
        matches2 = self.ratioTest(matches2)

        matches = self.symmetryTest(matches1, matches2)

        if(self.trainingImage and keypointsModel):
            cv2.drawMatches(frame, keypoints_frame, self.trainingImage, keypointsModel, matches, None)

    def fastRobustMatch(self, frame, descriptorsModel, keypointsModel):
        keypoints_frame, descriptors_frame = self.computeKeypointsAndDescriptors(frame)

        matches = self.descriptorMatcher.knnMatch(descriptors_frame.astype(np.uint8), descriptorsModel.astype(np.uint8), 2)
        matches = [x for (x,y) in self.ratioTest(matches)]

        if(self.trainingImage is not None and keypointsModel is not None):
            self.__imageMatching = cv2.drawMatches(frame, keypoints_frame, self.trainingImage, keypointsModel, matches, None)

        return matches, keypoints_frame



    