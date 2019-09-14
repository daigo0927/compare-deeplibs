import os, sys
sys.path.append(os.pardir)
import time
import numpy as np
import chainer
import chainer.functions as F
from chainer import iterators, optimizers
from chainer.dataset import concat_examples
from chainer.datasets import split_dataset_random
from chainer.backends.cuda import to_cpu

from model import ResNetMini
from dataset import Cifar10, Preprocess, Transform
from utils import prepare_parser, show_progress


def train(args):
    # Create dataset pipeline
    preprocess = Preprocess(resize_shape=args.resize_shape)
    transform = Transform(crop_shape=args.crop_shape,
                          rotate=args.rotate,
                          flip_left_right=args.flip_left_right,
                          flip_up_down=args.flip_up_down)
    dataset = Cifar10(dataset_dir=args.dataset_dir,
                      train_or_test='train',
                      preprocess=preprocess,
                      transform=transform)

    n_train = int(len(dataset)*(1-args.validation_split))
    tset, vset = split_dataset_random(dataset, n_train)
    titer = iterators.SerialIterator(tset, args.batch_size)
    viter = iterators.SerialIterator(vset, args.batch_size)
    
    device = chainer.get_device(args.device)

    # ------------- Build model and optimizer-----------------
    model = ResNetMini(args.filters, dataset.num_classes).to_device(device)
    optimizer = optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)

    # ------------- Actual training step-----------------
    n_batches = np.ceil(len(tset)/args.batch_size)
    for e in range(args.epochs):
        # ------------- Training --------------------
        dataset.train()
        i = 0
        while titer.epoch == e:
            batch = titer.next()
            images, labels = concat_examples(batch, device)
            start = time.time() # <<< Start time measurement ---
            logits = model(images)
            loss = F.softmax_cross_entropy(logits, labels)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            step_time = time.time() - start # --- Stop time measurement >>>

            if i%10 == 0: # --- Output logs -----
                acc = F.accuracy(logits, labels)
                show_progress(e+1, i+1, int(n_batches),
                              loss=to_cpu(loss.array),
                              accuracy=to_cpu(acc.array),
                              step_time=step_time)
            i += 1

        # ------------- Validation --------------------
        dataset.eval()
        losses, accs = [], []
        while viter.epoch == e:
            batch = viter.next()
            images, labels = concat_examples(batch, device)
            logits = model(images)
            loss = F.softmax_cross_entropy(logits, labels)
            acc = F.accuracy(logits, labels)
            losses.append(to_cpu(loss.array))
            accs.append(to_cpu(acc.array))
        print('\nValidation score: loss: {}, accuracy: {}.'\
              .format(np.mean(losses), np.mean(accs)))
        

if __name__ == '__main__':
    parser = prepare_parser()
    parser.add_argument('-g', '--gpu', dest='device',
                        type=int, nargs='?', default=-1,
                        help='GPU ID (negetive value indicates CPU) [-1]')
                        
    args = parser.parse_args()
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    train(args)
