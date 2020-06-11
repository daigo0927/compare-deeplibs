import torch
import torch.nn as nn
import argparse
import albumentations as A
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import ResNetMini
from dataset import prepare_samples, Cifar10


def accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    return (preds == labels).sum().item() / labels.size(0)


def train(dataset_dir, batch_size, learning_rate, epochs, num_workers):

    input_paths, target_labels = prepare_samples(dataset_dir, train_or_test='train')
    preprocess = A.Compose([A.Normalize()])
    augmentation = None
    dataset = Cifar10(input_paths, target_labels,
                      preprocess=preprocess, augmentation=augmentation)
    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=batch_size, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetMini(10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        running_acc = 0.0
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_acc += accuracy(logits, labels)
        print(f'Epoch: {epoch}, accuracy: {running_acc/(i+1)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', default='../cifar-10')
    parser.add_argument('-b', '--batch-size', default=32)
    parser.add_argument('-lr', '--learning-rate', default=0.001)
    parser.add_argument('-e', '--epochs', default=10)
    parser.add_argument('-nw', '--num-workers', default=0)
    args = parser.parse_args()
    
    train(**vars(args))

