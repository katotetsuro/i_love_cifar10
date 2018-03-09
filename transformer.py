import numpy as np
import random
import cv2
from chainercv import transforms

# original code from https://github.com/dsanno/chainer-cifar/blob/random_erasing/src/train.py


def random_erasing(x, offset_range=(-4, 5), s_range=(0.02, 0.4), r_width=3.0):
    image = np.zeros_like(x)
    size = x.shape[2]
    offset = np.random.randint(offset_range[0], offset_range[1], size=(2,))
    mirror = np.random.randint(2)
    remove = np.random.randint(2)
    top, left = offset
    left = max(0, left)
    top = max(0, top)
    right = min(size, left + size)
    bottom = min(size, top + size)
    if mirror > 0:
        x = x[:, :, ::-1]
    image[:, size-bottom:size-top, size-right:size -
          left] = x[:, top:bottom, left:right]
    if remove > 0:
        while True:
            s = np.random.uniform(s_range[0], s_range[1]) * size * size
            r = np.random.uniform(-np.log(r_width), np.log(r_width))
            r = np.exp(r)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, size)
            top = np.random.randint(0, size)
            if left + w < size and top + h < size:
                break

        # [katotetsuro] modify fill value
        c = np.random.rand(1)
        image[:, top:top + h, left:left + w] = c
    return image

# https://qiita.com/yu4u/items/70aa007346ec73b7ff05


def mixup(x, manual_blend=None):
    alpha = 0.2
    if not len(x) == 4:
        raise ValueError('mixup takes two images and two labels')

    l = np.random.beta(
        alpha, alpha, 1).astype(np.float32) if manual_blend == None else manual_blend
    x1, x2, *_ = x

    mixed_x = x1 * l + x2 * (1 - l)

    return mixed_x, l


class RandomErasingTransform(object):
    def __init__(self):
        print('RandomErasingあり')

    def __call__(self, in_data):
        x, y = in_data
        x = random_erasing(x)

        return x, y


class MixupTransform(object):
    def __init__(self, manual_blend=None, use_random_erasing=True):
        print('Mixupあり、RandomErasing{}'.format('あり' if use_random_erasing else 'なし'))
        self.manual_blend = manual_blend
        self.use_random_erasing = use_random_erasing

    def __call__(self, in_data):
        x, l = mixup(in_data, self.manual_blend)
        *_, y1, y2 = in_data

        if self.use_random_erasing:
            x = random_erasing(x)

        return x, (y1, y2, l)

def cv_rotate(img, angle):
    """
    https://github.com/mitmul/chainer-cifar10/blob/master/train.py
    """
    img = img.transpose(1, 2, 0)
    center = (img.shape[0] // 2, img.shape[1] // 2)
    r = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, r, img.shape[:2])
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    return img

class LessonTransform():
    def __init__(self, crop_size=(128, 128)):
        self.crop_size = crop_size

    def _add_noise(self, x, expand, angle, offset_range, s_range, r_width):
        ratio = random.uniform(1, expand)
        out_h, out_w = int(self.crop_size[0] * ratio), int(self.crop_size[1] * ratio)
        x = cv2.resize(x.transpose(1,2,0), (out_h, out_w))
        if x.shape[2] == 1:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        x = x.transpose((2,0,1))
        # Color augmentation
#            if self.pca_sigma != 0:
#                            x = transforms.pca_lighting(x, self.pca_sigma)
        # Random rotate
        angle = np.random.uniform(-angle, angle)
        x = cv_rotate(x, angle)

        # Random flip
        x = transforms.random_flip(x, x_random=True, y_random=True)
        # Random expand
        #if expand_ratio > 1:
        #    img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        x = transforms.random_crop(x, self.crop_size)
        x = random_erasing(x, offset_range, s_range, r_width)
        return x

    def __call__(self, in_data):
        x, t = in_data

        xt = x #self._add_noise(x.copy(), 1.1, 5, (-2, 2), (0.002, 0.04), 1.0)
        xs = self._add_noise(x.copy(), 2, 15, (-4, 5), (0.02, 0.4), 3.0)

        return xt, xs, t
