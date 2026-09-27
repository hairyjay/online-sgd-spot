import numpy as np
import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from . import shards

class CIFARShards(shards.Shards):
    class Net(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def __init__(self, args, pricing, drift):
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.classes = 10
        super().__init__(args, pricing, drift, classes=self.classes)
        if self.args.target == 0:
            self.args.target = 0.65
            print("DEFAULT -- setting target to {}".format(self.args.target))

    def testset(self):
        return datasets.CIFAR10(root='~/spot_aws/data',
                                            train=False,
                                            download=True,
                                            transform=self.test_transform)

    def trainset(self, idx=None):
        return datasets.CIFAR10(root='~/spot_aws/data',
                                            train=True,
                                            download=True,
                                            transform=self.train_transform), False

    def get_scheduler(self, parameters):
        if self.args.optimizer == 'adam':
            return optim.Adam(parameters, lr=self.args.lr, weight_decay=5e-4, betas=(0.9, 0.999), eps=1e-08), None
        else:
            return optim.SGD(parameters, lr=self.args.lr, momentum=0, weight_decay=5e-4), None
