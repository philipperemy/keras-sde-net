###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################

import argparse
from pathlib import Path

import numpy as np
import tensorflow.keras.backend as K

from sdenet.data import data_loader
from sdenet.eval import calculate_log
from sdenet.models import sdenet_mnist, resnet
from sdenet.models.helpers import load_weights


def main():
    parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
    parser.add_argument('--eva_iter', default=10, type=int, help='number of passes when evaluation')
    parser.add_argument('--network', type=str, choices=['resnet', 'sdenet', 'mc_dropout'], default='resnet')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--dataset', required=True, help='in domain dataset')
    parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
    parser.add_argument('--out_dataset', required=True, help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
    parser.add_argument('--pre_trained_net', default=None, help="path to pre trained_net h5")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    outf = Path('test') / args.network
    outf.mkdir(parents=True, exist_ok=True)

    print('Load model')
    if args.network == 'resnet':
        model = resnet.ResidualNet()
        args.eva_iter = 1
    elif args.network == 'sdenet':
        model = sdenet_mnist.SDENet_mnist(layer_depth=6, num_classes=10, dim=64)
    else:
        raise Exception('Model not found.')
    # elif args.network == 'mc_dropout':
    #     model = models.Resnet_dropout()

    if args.pre_trained_net is not None and Path(args.pre_trained_net).exists():
        model(np.ones(shape=(12, 28, 28, 1)))  # forward pass to compute input shapes.
        load_weights(model, args.pre_trained_net)

    print('load target data: ', args.dataset)
    _, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)

    print('load non target data: ', args.out_dataset)
    nt_train_loader, nt_test_loader = data_loader.getDataSet(args.out_dataset, args.batch_size, args.test_batch_size,
                                                             args.imageSize)

    def generate_target():
        correct = 0
        total = 0
        f1 = open('%s/confidence_Base_In.txt' % outf, 'w')
        f3 = open('%s/confidence_Base_Succ.txt' % outf, 'w')
        f4 = open('%s/confidence_Base_Err.txt' % outf, 'w')

        for data, targets in test_loader:
            data, targets = np.array(data), np.array(targets)
            total += len(data)
            data = np.transpose(data, (0, 2, 3, 1))
            batch_output = 0
            for j in range(args.eva_iter):
                batch_output = batch_output + K.softmax(model(data), axis=-1)
            batch_output = batch_output / args.eva_iter
            # compute the accuracy
            predicted = K.argmax(batch_output, axis=1).numpy()
            correct += (predicted == targets).sum()
            correct_index = (predicted == targets)
            for i in range(len(data)):
                # confidence score: max_y p(y|x)
                # output = batch_output[i].view(1, -1)
                output = K.expand_dims(batch_output[i], axis=0)
                soft_out = K.max(output).numpy()
                f1.write("{}\n".format(soft_out))
                if correct_index[i] == 1:
                    f3.write("{}\n".format(soft_out))
                elif correct_index[i] == 0:
                    f4.write("{}\n".format(soft_out))
        f1.close()
        f3.close()
        f4.close()
        print('\n Final Accuracy: {}/{} ({:.2f}%)\n '.format(correct, total, 100. * correct / total))

    def generate_non_target():
        total = 0
        f2 = open('%s/confidence_Base_Out.txt' % outf, 'w')
        for data, targets in nt_test_loader:
            data, targets = np.array(data), np.array(targets)
            data = np.transpose(data, (0, 2, 3, 1))
            total += len(data)
            batch_output = 0
            for j in range(args.eva_iter):
                batch_output = batch_output + K.softmax(model(data), axis=-1)
            batch_output = batch_output / args.eva_iter
            for i in range(len(data)):
                # confidence score: max_y p(y|x)
                output = K.expand_dims(batch_output[i], axis=0)
                soft_out = K.max(output).numpy()
                f2.write("{}\n".format(soft_out))
        f2.close()

    print('generate log from in-distribution data')
    generate_target()
    print('generate log  from out-of-distribution data')
    generate_non_target()
    print('calculate metrics for OOD')
    calculate_log.metric(outf, 'OOD')
    print('calculate metrics for mis')
    calculate_log.metric(outf, 'mis')


if __name__ == '__main__':
    main()
