from __future__ import print_function
import argparse
from os.path import join

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import transformer
import dataset
import numpy as np
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


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

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

    predictor = shaked_pyramid_net.PyramidNet(skip=True)
    model = L.Classifier(predictor, lossfun=soft_label_classification_loss,
                         accfun=soft_label_classification_acc)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # augment train data
    train = dataset.PairwiseCifar10((train, None))
    train = chainer.datasets.transform_dataset.TransformDataset(
        train, transformer.MixupTransform())
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
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

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
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
