# コンペが終わるまでgitignoreされる秘密のローダー
from pathlib import Path
import random
import chainer
from chainercv import transforms
import numpy as np
import pandas as pd
import cv2

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

class FixedSizeDataset(chainer.datasets.LabeledImageDataset):
    def __init__(self, pairs, root='.', train=True, random_angle=15., pca_sigma=25.0, expand_ratio=1.0,
        crop_size=(128, 128)):
        super().__init__(pairs, root=root)
        self.train = train
        self.random_angle = random_angle
        self.pca_sigma = pca_sigma
        self.expand_ratio = expand_ratio
        self.crop_size = crop_size

    def get_example(self, index):
        x, y = super().get_example(index)
        # random_expandをする代わりに、最初に target_size * random_expandの大きさにリサイズしておいて、
        # あとでrandom croppingすればいいのではないかと思いました
        ratio = random.uniform(1, self.expand_ratio)
        out_h, out_w = int(self.crop_size[0] * ratio), int(self.crop_size[1] * ratio)
        x = cv2.resize(x.transpose(1,2,0), (out_h, out_w))
#        x = cv2.resize(x.transpose(1,2,0), (128, 128))

        if x.shape[2] == 1:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        x = x.transpose((2,0,1))

        if self.train:
            # Color augmentation
#            if self.pca_sigma != 0:
#                            x = transforms.pca_lighting(x, self.pca_sigma)
            # Random rotate
            if self.random_angle != 0:
                angle = np.random.uniform(-self.random_angle, self.random_angle)
                x = cv_rotate(x, angle)

            # Random flip
            x = transforms.random_flip(x, x_random=True, y_random=True)
            # Random expand
            #if expand_ratio > 1:
            #    img = transforms.random_expand(img, max_ratio=expand_ratio)
            # Random crop
            x = transforms.random_crop(x, self.crop_size)

        x /= 255.0
        return x, y

class CookPadLoader(chainer.dataset.DatasetMixin):
    """
    mixupをするときに使うdataset
    create_database.ipynbでtrain.txtとtest.txtを作成している想定
    """
    def __init__(self, base_dir, file_name):

        self.gen = FixedSizeDataset(str(Path(base_dir).joinpath(file_name)), root=base_dir)
        self.length = int(len(self.gen)/2)

    def __len__(self):
        return self.length

    def get_example(self, index):
        if index >= self.length:
            raise IndexError()

        i = 2 * index
        j = 2 * index + 1
        x1, y1 = self.gen.get_example(i)
        x2, y2 = self.gen.get_example(j)
        return x1, x2, y1, y2

class Resizing():
    """
    ターゲットサイズにリサイズするんだけど、アスペクト比を固定すべきかどうか悩むなー
    """
    def __call__(self, in_data):
        x, y = in_data
        x = x.transpose(1,2,0)
        w, h, _ = x.shape
        
        x = cv2.resize(x.transpose(1,2,0), (128, 128))
        return x, y
