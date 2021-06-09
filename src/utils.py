import cv2

def createFeatures(featureName, numKeypoints):
    if featureName == "ORB":
        detector = cv2.ORB_create(nfeatures=numKeypoints)
        descriptor = cv2.ORB_create(nfeatures=numKeypoints)
        return detector, descriptor
    else:
        raise NotImplementedError


def createMatcher(featureName, useFLANN):
    if featureName == "ORB" and useFLANN is True:
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
            table_number = 6, # 12
            key_size = 12,     # 20
            multi_probe_level = 1) #2
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    elif featureName == "ORB" and useFLANN is False:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    else:
        raise NotImplementedError
