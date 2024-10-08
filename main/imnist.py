import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import v2
from kornia.morphology import erosion, dilation

from . import shards

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

    def __init__(self, args, pricing):
        super().__init__(args, pricing)
        if self.args.target == 0:
            self.args.target = 0.85
            print("DEFAULT -- setting target to {}".format(self.args.target))
        self.norm_mean = 0.1307
        self.norm_std = 0.3081
        self.norm_min = -0.42421296
        self.train_transform = transforms.Compose([
                               transforms.PILToTensor(),
                               v2.Lambda(self.rand_thicken),
                               v2.ElasticTransform(alpha=30.0, sigma=3.0),
                               transforms.RandomPerspective(),
                               transforms.RandomAffine(30, translate=(0.1, 0.1)),
                               transforms.Normalize((self.norm_mean,), (self.norm_std,)),
                               v2.Lambda(self.fill_nan)
        ])
        self.test_transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((self.norm_mean,), (self.norm_std,)),
                               v2.Lambda(self.fill_nan)
        ])

    def testset(self):
        return datasets.MNIST(root='~/spot_aws/data',
                                        train=False,
                                        download=True,
                                        transform=self.test_transform)

    def trainset(self, idx):
        return datasets.MNIST(root='~/spot_aws/data',
                                        train=True,
                                        download=True,
                                        transform=self.train_transform), False

    def rand_thicken(self, image:torch.Tensor) -> torch.Tensor:
        image = torch.unsqueeze(image.float(), 0)
        t = torch.randint(1, 3, (2,))
        kernel = torch.ones((t[0], t[1]))
        if np.random.rand() < 0.5:
            return torch.squeeze(erosion(image, kernel=kernel), 0)
        else:
            return torch.squeeze(dilation(image, kernel=kernel), 0)
        
    def fill_nan(self, image:torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(image, nan=self.norm_min)
