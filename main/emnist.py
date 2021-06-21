import numpy as np
import numpy.random as random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import shards

class EMNISTShards(shards.Shards):
    '''
    class Net(nn.Module):
        """
        VGG-5 Model
        Based on - https://github.com/kkweon/mnist-competition
        from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
        """
        def two_conv_pool(self, in_channels, f1, f2):
            s = nn.Sequential(
                nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(f1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(f2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            for m in s.children():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            return s

        def three_conv_pool(self,in_channels, f1, f2, f3):
            s = nn.Sequential(
                nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(f1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(f2),
                nn.ReLU(inplace=True),
                nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(f3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            for m in s.children():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            return s


        def __init__(self, num_classes=62):
            super().__init__()
            self.l1 = self.two_conv_pool(1, 64, 64)
            self.l2 = self.two_conv_pool(64, 128, 128)
            self.l3 = self.three_conv_pool(128, 256, 256, 256)
            self.l4 = self.three_conv_pool(256, 256, 256, 256)

            self.classifier = nn.Sequential(
                nn.Dropout(p = 0.5),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p = 0.5),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
            x = self.l4(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return F.log_softmax(x, dim=1)
    '''


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 47)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    def __init__(self, args, pricing):
        super().__init__(args, pricing)
        self.train_transform = transforms.Compose([
                               torchvision.transforms.RandomPerspective(),
                               torchvision.transforms.RandomRotation(10, fill=(0,)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
        ])
        self.test_transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
        ])

    def testset(self):
        return torchvision.datasets.EMNIST(root='~/spot_aws/data',
                                            split='bymerge',
                                            train=False,
                                            download=True,
                                            transform=self.test_transform)

    def trainset(self):
        return torchvision.datasets.EMNIST(root='~/spot_aws/data',
                                            split='bymerge',
                                            train=True,
                                            download=True,
                                            transform=self.train_transform)
