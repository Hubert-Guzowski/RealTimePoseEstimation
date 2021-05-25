import os
from glob import glob

import numpy as np
import cv2


def crop_image(image, h_factor, w_factor):
    """ Reduce image height by h_factor from up and down.
    Analogously for image width. """

    h, w = image.shape[:2]
    image = image[
            int(h_factor * h):-int(h_factor * h),
            int(w_factor * w):-int(w_factor * w)
            ]
    return image


def load_images(directory, target_shape=None, crop=True):
    """ Load images from directory and return array of stacked images. """

    paths = glob(os.path.join(directory, '*.JPG'))
    images = []
    for path in paths:
        img = cv2.imread(path)
        if target_shape is not None:
            img = cv2.resize(img, dsize=target_shape)
        if crop:
            img = crop_image(img, 0.1, 0.1)
        images.append(img)
    return np.stack(images)


def visualize_dataset(image_dataset):
    for image in image_dataset:
        cv2.imshow('image', image)
        cv2.waitKey()


if __name__ == '__main__':
    DATA_DIR = '../tea_box/tracking'

    dataset = load_images(DATA_DIR, target_shape=(640, 480))
    print(f'Loaded images shape: {dataset.shape}')
    visualize_dataset(dataset)
