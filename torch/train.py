import os, sys
sys.path.append(os.pardir)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import ResNetMini
from dataset import Cifar10
from utils import prepare_parser, show_progress


def accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    return (preds == labels).sum().item() / labels.size(0)


def train(args):
    # Create preprocesses
    preps = []
    if args.resize_shape:
        preps.append(transforms.Resize(args.resize_shape))
    preps += [
        lambda x: np.asarray(x),
        lambda x: x/255.0,
        transforms.ToTensor()
    ]
    preprocess = transforms.Compose(preps)
    
    # Create augumentations
    trans = []
    if args.crop_shape:
        trans.append(transforms.RandomCrop(args.crop_shape))
    if args.rotate:
        trans.append(transforms.RandomRotate(45))
    if args.flip_left_right:
        trans.append(transforms.RandomFlipLeftRight())
    if args.flip_up_down:
        trans.append(transforms.RandomFlipUpDown())
    transform = transforms.Compose(trans)

    # ------------- Build dataset pipeline -----------------
    dataset = Cifar10(dataset_dir=args.dataset_dir,
                      train_or_test='train',
                      preprocess=preprocess,
                      transform=transform)

    # Build dataloader
    n_train = int(len(dataset)*(1-args.validation_split))
    n_val = len(dataset) - n_train
    tset, vset = random_split(dataset, [n_train, n_val])
    tloader = DataLoader(tset, args.batch_size, shuffle=True, num_workers=1)
    vloader = DataLoader(vset, args.batch_size, shuffle=False, num_workers=1)

    # ------------- Build model and optimizer-----------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNetMini(args.filters,
                       dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # ------------- Training loop -----------------
    start_loop = time.time()
    for e in range(args.epochs):
        start_epoch = time.time()
        
        model.train()
        tset.dataset.train()
        for i, (images, labels) in enumerate(tloader):
            start_batch = time.time()
            images, labels = images.to(device, torch.float32), labels.to(device)
            model.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_time = time.time() - start_batch

            if i%args.check_step == 0 or i+1 == len(tloader): # ----- Output log -----
                acc = accuracy(logits, labels)
                show_progress(e+1, i+1, len(tloader),
                              loss=loss.item(), accuracy=acc,
                              batch_time=batch_time)

        train_time = time.time() - start_epoch
        print('\nTraining time: {}.'.format(train_time))

        # ------------- Evaluation -----------------
        model.eval()
        vset.dataset.eval()
        losses, accs = [], []
        for images, labels in vloader:
            images, labels = images.to(device, torch.float32), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            acc = accuracy(logits, labels)
            losses.append(loss.item())
            accs.append(acc)

        val_time = time.time() - start_epoch - train_time
        print('Validation score: loss: {}, accuracy: {}, time: {}.'\
              .format(np.mean(losses), np.mean(accs), val_time))

        epoch_time = time.time() - start_epoch
        print('Epoch time: {}sec'.format(epoch_time))
        print()

    loop_time = time.time() - start_loop
    print('Total time: {}sec.'.format(loop_time))


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    train(args)
