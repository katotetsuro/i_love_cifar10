import warnings
import numpy as np
from PIL import Image
import Augmentor


class AugmentorTransform():
    def __init__(self, scale=1.0 / 255, size=128, train=True):
        self.p = Augmentor.Pipeline()
        self.p.resize(probability=1, width=size, height=size,
                      resample_filter='BILINEAR')
        if train:
            self.p.rotate(probability=0.5, max_left_rotation=10,
                          max_right_rotation=10)
            self.p.flip_left_right(probability=0.5)
            self.p.flip_top_bottom(probability=0.5)
            self.p.zoom_random(probability=0.5, percentage_area=0.8)
            self.p.random_distortion(
                probability=0.5, grid_width=4, grid_height=4, magnitude=5)
            self.p.random_erasing(probability=0.5, rectangle_area=0.5)
        self.scale = scale

    def __call__(self, in_data):
        x, t = in_data

        if x.dtype == np.uint8:
            pass
        elif x.dtype == np.float32:
            if np.max(x) < 1.0:
                warnings.warn(
                    'scale is [0, 1]? AugmentorTransform assumes [0, 255]')
            x = x.astype(np.uint8)
        else:
            raise ValueError('cannot handle dtype {}'.format(x.dtype))

        img = Image.fromarray(x.transpose((1, 2, 0)))
        for operation in self.p.operations:
            img = operation.perform_operation([img])[0]

        img = np.array(img, dtype=np.float32) * self.scale
        img = img.transpose((2, 0, 1))
        return img, t
