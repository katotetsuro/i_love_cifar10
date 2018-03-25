# https://github.com/dsanno/chainer-cifar/blob/random_erasing/src/net.py
import math
import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainer import function
from chainer import link
from chainer.utils import array
from chainer.utils import type_check


class ShakeNoiseMultiplier(function.Function):

    """forwardとbackwardで異なる乱数が乗算される"""

    def __init__(self, forward_range, backward_range):
        self.forward_range = forward_range
        self.backward_range = backward_range

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(
            in_types[0].dtype.kind == 'f'
        )

    def forward(self, inputs):
        x1, = inputs
        xp = cuda.get_array_module(x1)
        x_shape = x1.shape
        w_shape = [x_shape[0]] + [1] * (len(x_shape) - 1)
        if xp == np:
            weight = xp.random.rand(*w_shape).astype(x1.dtype)
        else:
            weight = xp.random.rand(*w_shape, dtype=x1.dtype)

        a, b = self.forward_range
        weight = weight * (b - a) + a
        return x1 * weight,

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*gy)
        g_shape = gy[0].shape
        w_shape = [g_shape[0]] + [1] * (len(g_shape) - 1)
        if xp == np:
            weight = xp.random.rand(*w_shape).astype(gy[0].dtype)
        else:
            weight = xp.random.rand(*w_shape, dtype=gy[0].dtype)

        a, b = self.backward_range
        weight = weight * (b - a) + a
        return gy[0] * weight,


def shake_noise_multiplier(x, forward_range, backward_range):
    return ShakeNoiseMultiplier(forward_range, backward_range)(x)


class BatchConv2D(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize, stride=1, pad=0, activation=F.relu):
        super(BatchConv2D, self).__init__(
            conv=L.Convolution2D(ch_in, ch_out, ksize, stride, pad),
            bn=L.BatchNormalization(ch_out),
        )
        self.activation = activation

    def __call__(self, x):
        h = self.bn(self.conv(x))
        if self.activation is None:
            return h
        return self.activation(h)


class PyramidBlock(chainer.Chain):
    def __init__(self, ch_in, ch_out, stride=1, activation=F.relu, skip_ratio=0, alpha=(-1, 1), beta=(0, 1)):
        initializer = initializers.Normal(
            scale=math.sqrt(2.0 / (ch_out * 3 * 3)))
        super(PyramidBlock, self).__init__(
            conv1=L.Convolution2D(ch_in, ch_out, 3, stride,
                                  1, initialW=initializer),
            conv2=L.Convolution2D(ch_out, ch_out, 3, 1,
                                  1, initialW=initializer),
            bn1=L.BatchNormalization(ch_in),
            bn2=L.BatchNormalization(ch_out),
            bn3=L.BatchNormalization(ch_out),
        )
        self.activation = activation
        # skip = shake shakeを適用する解釈する
        # skip_ratio はshake shakeを適用する確率
        self.skip_ratio = skip_ratio
        self.alpha = alpha
        self.expect_alpha = (alpha[0] + alpha[1]) * 0.5
        self.beta = beta
        self.expect_beta = (beta[0] + beta[1]) * 0.5

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x.data)
        sh, sw = self.conv1.stride
        c_out, c_in, kh, kw = self.conv1.W.data.shape
        b, c, hh, ww = x.data.shape
        if sh == 1 and sw == 1:
            shape_out = (b, c_out, hh, ww)
        else:
            hh = (hh + 2 - kh) // sh + 1
            ww = (ww + 2 - kw) // sw + 1
            shape_out = (b, c_out, hh, ww)
        h = x
        if x.data.shape[2:] != shape_out[2:]:
            x = F.average_pooling_2d(x, 1, 2)
        if x.data.shape[1] != c_out:
            n, c, hh, ww = x.data.shape
            pad_c = c_out - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p)
            x = F.concat((x, p), axis=1)

        h = self.bn1(h)
        h = self.conv1(h)
        h = self.bn2(h)
        if self.activation is not None:
            h = self.activation(h)
        h = self.conv2(h)
        h = self.bn3(h)

        # 再掲：skip==True -> shakeする
        if not chainer.config.train:
            skip = False
            scale = (1 - self.skip_ratio) + self.expect_alpha * self.skip_ratio
            return h * scale + x
        else:
            skip = np.random.rand() < self.skip_ratio

        if skip:
            return shake_noise_multiplier(h, self.alpha, self.beta) + x
        else:
            return h + x


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h


class PyramidNet(chainer.Chain):
    def __init__(self, depth=18, alpha=16, start_channel=16, skip=False, num_class=10, pooling=F.max_pooling_2d):
        super(PyramidNet, self).__init__()

        with self.init_scope():
            channel_diff = float(alpha) / depth
            channel = start_channel
            links = [('bconv1', BatchConv2D(3, channel, 3, 1, 1))]
            skip_size = depth * 3 - 3
            for i in six.moves.range(depth):
                if skip:
                    skip_ratio = float(i) / skip_size * 0.5
                else:
                    skip_ratio = 0
                in_channel = channel
                channel += channel_diff
                links.append(('py{}'.format(len(links)), PyramidBlock(
                    int(round(in_channel)), int(round(channel)),  skip_ratio=skip_ratio)))
            in_channel = channel
            channel += channel_diff
            links.append(('py{}'.format(len(links)), PyramidBlock(
                int(round(in_channel)), int(round(channel)), stride=2)))
            if pooling is not None:
                links.append(('_pooling{}'.format(len(links)), lambda x: F.max_pooling_2d(
                    x, ksize=(2, 2), stride=2, pad=0)))
            for i in six.moves.range(depth - 1):
                if skip:
                    skip_ratio = float(i + depth) / skip_size * 0.5
                else:
                    skip_ratio = 0
                in_channel = channel
                channel += channel_diff
                links.append(('py{}'.format(len(links)), PyramidBlock(
                    int(round(in_channel)), int(round(channel)),  skip_ratio=skip_ratio)))
            in_channel = channel
            channel += channel_diff
            links.append(('py{}'.format(len(links)), PyramidBlock(
                int(round(in_channel)), int(round(channel)), stride=2)))
            if pooling is not None:
                links.append(('_pooling{}'.format(len(links)), lambda x: F.max_pooling_2d(
                    x, ksize=(2, 2), stride=2, pad=0)))
            for i in six.moves.range(depth - 1):
                if skip:
                    skip_ratio = float(i + depth * 2 - 1) / skip_size * 0.5
                else:
                    skip_ratio = 0
                in_channel = channel
                channel += channel_diff
                links.append(('py{}'.format(len(links)), PyramidBlock(
                    int(round(in_channel)), int(round(channel)),  skip_ratio=skip_ratio)))
            links.append(('bn{}'.format(len(links)),
                          L.BatchNormalization(int(round(channel)))))
            links.append(('_relu{}'.format(len(links)), F.relu))
            # attempt to global average pooling
            links.append(('conv{}'.format(len(links)), L.Convolution2D(
                None, num_class, ksize=3, stride=1, pad=0)))
            links.append(('_apool{}'.format(len(links)),
                          _global_average_pooling_2d))

            for name, f in links:
                if not name.startswith('_'):
                    self.add_link(*(name, f))
            self.layers = links

    def __call__(self, x):
        h = x
        for name, f in self.layers:
            h = f(h)
        return h
