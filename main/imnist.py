import numpy as np
import numpy.random as random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from . import shards
from infimnist import InfiMNIST

class InfiMNISTShards(shards.Shards):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    class InfiMNISTTrain(InfiMNIST):
        def __init__(self, idx, N, train=True, num_transformations=1, transform=None, target_transform=None):
            self.N = N
            self.idx = idx
            self.is_infinite = True
            super().__init__(train, num_transformations, transform, target_transform)


    def __init__(self, args, pricing):
        super().__init__(args, pricing)
        if self.args.target == 0:
            self.args.target = 0.85
            print("DEFAULT -- setting target to {}".format(self.args.target))
        self.train_transform = transforms.Compose([
                               torchvision.transforms.RandomPerspective(),
                               torchvision.transforms.RandomAffine(30, translate=(0.1, 0.1)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    def testset(self):
        return torchvision.datasets.MNIST(root='~/spot_aws/data',
                                        train=False,
                                        download=True,
                                        transform=self.test_transform)

    def trainset(self, idx):
        return self.InfiMNISTTrain(idx, self.args.size,
                                train=True,
                                num_transformations=4,
                                transform=self.train_transform,
                                target_transform=None), True
