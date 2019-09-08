import os, sys
sys.path.append(os.pardir)
import time
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions, triggers

from model import ResNetMini
from dataset import Cifar10
from utils import prepare_parser, show_progress


def train(args):
    # TODO: dataset definition
    
    device = chainer.get_device(args.device)

    # ------------- Build model and optimizer-----------------
    net = ResNetMini(args.filters, dataset.num_classes)
    model = L.Classifier(net)
    model.to_device(device)

    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup()

    # TODO: training setup


if __name__ == '__main__':
    parser = prepare_parser()
    parser.add_argument('-g', '--gpu', dest='device',
                        type=int, nargs='?', default=-1,
                        help='GPU ID (negetive value indicates CPU) [-1]')
                        
    args = parser.parse_args()
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    train(args)
