import numpy as np

# original code from https://github.com/dsanno/chainer-cifar/blob/random_erasing/src/train.py


def random_erasing(x):
    image = np.zeros_like(x)
    size = x.shape[2]
    offset = np.random.randint(-4, 5, size=(2,))
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
            s = np.random.uniform(0.02, 0.4) * size * size
            r = np.random.uniform(-np.log(3.0), np.log(3.0))
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
