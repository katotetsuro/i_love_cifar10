from __future__ import print_function
import argparse
from os.path import join
import random
import copy

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
import mean_teacher_train_chain

class ResNet(chainer.links.ResNet50Layers):
    def __call__(self, x):
        return super().__call__(x, layers=['fc6'])['fc6']


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
    parser.add_argument('--learnrate', '-l', type=float, default=0.01,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--aug_method', '-a', default='random_erasing', choices=['none', 'mixup', 'random_erasing', 'both'],
                        help='data augmentation strategy')
    parser.add_argument('--model', '-m', default='pyramid', choices=['resnet50', 'pyramid'],
                        help='data augmentation strategy')
    parser.add_argument('--weights', '-w', default='', help='initial weight')
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
        # trainのうち10000枚を検証用にとっておく. splitと呼ぶ
        # testの10000枚はラベルを-1に変換して、ラベルなしのデータとして扱う. unlabeledと呼ぶ
        # 1. testに対して、精度があがるのか?
        # 2. splitで、精度の向上と連動した様子が観察できるのか？
        train, test = get_cifar10()
        split = train[-10000:]
        train = train[:-10000]
        # label = -1のデータとして扱う
        unlabeled = [(x[0], -1) for x in test]
        print(f'train:{len(train)}, unlabeled:{len(unlabeled)}, test:{len(test)}')
        train = chainer.datasets.ConcatenatedDataset(train, unlabeled)

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

    if not args.weights == '':
        print(f'loading weights from {args.weights}')
        chainer.serializers.load_npz(args.weights, predictor)

    model = mean_teacher_train_chain.MeanTeacherTrainChain(model=predictor)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # augment train data
    print('currently, aug_method is ignored')
    train = dataset.SingleCifar10((train, None))
    train = chainer.datasets.transform_dataset.TransformDataset(
            train, transformer.LessonTransform(crop_size=(32, 32)))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, shuffle=True)
    split_iter = chainer.iterators.SerialIterator(split, args.batchsize,
                                                 repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # teacherをupdateするためのextension
    def update_teacher(trainer):
        model.on_update_finished(trainer)
    trainer.extend(update_teacher)

    # Evaluate the model with the test dataset for each epoch
    eval_trigger = (1, 'epoch')
    classifier = chainer.links.Classifier(model.teacher)
    split_evaluator = extensions.Evaluator(split_iter, classifier, device=args.gpu)
    split_evaluator.name = 'observable_validation'
    trainer.extend(split_evaluator, trigger=eval_trigger)

    truth_evaluator = extensions.Evaluator(test_iter, classifier, device=args.gpu)
    truth_evaluator.name = 'truth_validation'
    trainer.extend(truth_evaluator, trigger=eval_trigger)

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
        ['epoch', 'lr', 'main/class_loss', 'main/consistency_loss', 'main/loss',
         'observable_validation/main/loss', 'observable_validation/main/accuracy',
          'truth_validation/main/accuracy', 'truth_validation/main/loss', 'elapsed_time']))

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
