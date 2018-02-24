from __future__ import print_function
import argparse
from os.path import join
import random

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100
from chainer.links import ResNet50Layers

from chainerui.utils import save_args
from chainerui.extensions import CommandsExtension

import transformer
import dataset
import shaked_pyramid_net


def soft_label_classification_loss(x, t):
    xp = chainer.cuda.get_array_module(t)
    if t.shape == (len(x), 3):
        y1, y2, w = np.split(t, (1, 2), axis=1)
        w = w.reshape((-1,))
        y1 = y1.astype(xp.int32).reshape((-1, ))
        y2 = y2.astype(xp.int32).reshape((-1, ))
        loss = F.matmul(w, F.softmax_cross_entropy(x, y1, reduce='no')) + \
            F.matmul((1-w), F.softmax_cross_entropy(x, y2, reduce='no'))
        loss /= len(x)
        return loss
    else:
        return F.softmax_cross_entropy(x, t)

# 2つの画像がまざったサンプルに対する精度ってよくわかんないし、
# まーval accが上がればいいしなってことで適当に作った


def soft_label_classification_acc(x, t):
    xp = chainer.cuda.get_array_module(t)
    if t.shape == (len(x), 3):
        dominant = xp.max(t[:, :2], axis=1).astype(xp.int32)
        return F.accuracy(x, dominant)
    else:
        return F.accuracy(x, t)


class ResNet(chainer.links.ResNet50Layers):
    def __call__(self, x):
        return super().__call__(x)['prob']


def set_random_seed(seed):
    """
    https://qiita.com/TokyoMickey/items/cc8cd43545f2656b1cbd
    """

    # set Python random seed
    random.seed(seed)

    # set NumPy random seed
    np.random.seed(seed)

    # set Chainer(CuPy) random seed
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='seed for random values')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.1,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--aug_method', '-a', default='both', choices=['none', 'mixup', 'random_erasing', 'both'],
                        help='data augmentation strategy')
    parser.add_argument('--model', '-m', default='pyramid', choices=['resnet50', 'pyramid'],
                        help='data augmentation strategy')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print(args)
    print('')

    set_random_seed(args.seed)

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    if args.model == 'resnet50':
        predictor = ResNet(None)
        predictor.fc6 = L.Linear(2048, class_labels)
    elif args.model == 'pyramid':
        predictor = shaked_pyramid_net.PyramidNet(skip=True)

    # 下の方にあるtrain dataのtransformの条件分岐とかぶってるけどなー
    if args.aug_method in ('both', 'mixup'):
        lossfun = soft_label_classification_loss
        accfun = soft_label_classification_acc
    else:
        lossfun = F.softmax_cross_entropy
        accfun = F.accuracy

    model = L.Classifier(predictor, lossfun=lossfun, accfun=accfun)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # augment train data
    if args.aug_method == 'none':
        print('data augmentationなしです')
        train = dataset.SingleCifar10((train, None))
    elif args.aug_method in ('both', 'mixup'):
        use_random_erasing = args.aug_method == 'both'
        train = dataset.PairwiseCifar10((train, None))
        train = chainer.datasets.transform_dataset.TransformDataset(
            train, transformer.MixupTransform(use_random_erasing=use_random_erasing))
    elif args.aug_method == 'random_erasing':
        train = dataset.SingleCifar10((train, None))
        train = chainer.datasets.transform_dataset.TransformDataset(
            train, transformer.RandomErasingTransform())

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    eval_trigger = (1, 'epoch')
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        device=args.gpu), trigger=eval_trigger)

    # Reduce the learning rate by half every 25 epochs.
    lr_drop_epoch = [int(args.epoch*0.5), int(args.epoch*0.75)]
    lr_drop_ratio = 0.1
    print(f'lr schedule: {lr_drop_ratio}, timing: {lr_drop_epoch}')

    def lr_drop(trainer):
        trainer.updater.get_optimizer('main').lr *= lr_drop_ratio
    trainer.extend(
        lr_drop,
        trigger=chainer.training.triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch'))
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'lr', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())
    # interact with chainerui
    trainer.extend(CommandsExtension(), trigger=(100, 'iteration'))
    # save args
    save_args(args, args.out)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
