# まったく本質的じゃないんだけど、この後自前のデータセットで学習したいので
# cifar10みたいにtuple datasetじゃなくて、(画像1, 画像2, ラベル1, ラベル2）のデータがとれる mixinにする

import chainer
import numpy as np


class PairwiseCifar10(chainer.dataset.DatasetMixin):

    names = ['airplane',
             'automobile',
             'bird',
             'cat',
             'deer ',
             'dog  ',
             'frog ',
             'horse',
             'ship ',
             'truck']

    def __init__(self, dataset):
        train, _ = dataset
        self.x = np.stack([x[0] for x in train])
        self.y = np.stack([x[1] for x in train])
        self.num_samples = int(len(self.y) / 2)
        self.num_cats = len(self.names)

    def __len__(self):
        return self.num_samples

    def get_example(self, index):
        if index >= self.num_samples:
            raise IndexError()
        i = 2*index
        j = 2*index+1

#        yi = np.zeros(self.num_cats)
#        yi[self.y[i]] = 1
#        yj = np.zeros(self.num_cats)
#        yj[self.y[j]] = 1

        ret = (self.x[i], self.x[j], self.y[i], self.y[j])

        if j == self.num_samples:
            print('データをシャッフルします')
            indices = np.arange(len(self.y))
            np.random.shuffle(indices)
            self.x = np.take(self.x, indices, axis=0)
            self.y = np.take(self.y, indices, axis=0)

        return ret


class SingleCifar10(chainer.dataset.DatasetMixin):

    names = ['airplane',
             'automobile',
             'bird',
             'cat',
             'deer ',
             'dog  ',
             'frog ',
             'horse',
             'ship ',
             'truck']

    def __init__(self, dataset):
        train, _ = dataset
        self.x = np.stack([x[0] for x in train])
        self.y = np.stack([x[1] for x in train])
        self.num_samples = len(self.y)
        self.num_cats = len(self.names)

    def __len__(self):
        return self.num_samples

    def get_example(self, index):
        if index >= self.num_samples:
            raise IndexError()

        return self.x[index], self.y[index]
