import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchvision.models as models
from kornia.morphology import erosion, dilation

from . import shards

class ImageResNetShards(shards.Shards):
    class Net(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, classes)

        def forward(self, x):
            return self.model(x)

    def __init__(self, args, pricing, drift):
        self.classes = 1000
        super().__init__(args, pricing, drift, classes=self.classes)
        if self.args.target == 0:
            self.args.target = 0.90
            print("DEFAULT -- setting target to {}".format(self.args.target))
        # self.norm_mean = 0.1307
        # self.norm_std = 0.3081
        # self.norm_min = -0.42421296
        self.train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                                transforms.RandomHorizontalFlip(0.5),
                                v2.ElasticTransform(alpha=30.0, sigma=3.0),
                                v2.RandomPerspective(),
                                v2.RandomAffine(30, translate=(0.1, 0.1)),
                                transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(size=256, antialias=True),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def testset(self):
        return datasets.ImageFolder(root='~/spot_aws/data/ImageNet/val',
                                            transform=self.test_transform)

    def trainset(self, idx=None):
        return datasets.ImageFolder(root='~/spot_aws/data/ImageNet/train',
                                            transform=self.train_transform), True

    # def get_test_augment(self):
    #     return torch.nn.Sequential(
    #         transforms.Resize(size=256, antialias=True),
    #         transforms.CenterCrop(224),
    #         v2.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
    #         v2.Lambda(self.fill_nan)
    #     )
    
    # def get_train_augment(self):
    #     return torch.nn.Sequential(
    #         transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    #         transforms.RandomHorizontalFlip(0.5),
    #         v2.ElasticTransform(alpha=30.0, sigma=3.0),
    #         v2.RandomPerspective(),
    #         v2.RandomAffine(30, translate=(0.1, 0.1)),
    #         v2.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
    #         v2.Lambda(self.fill_nan)
    #     )

    def get_scheduler(self, parameters):
        optimizer = optim.SGD(parameters, lr=0.007, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.175,
            total_steps=30*1281167,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e-4
        )
        return optimizer, scheduler
    

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
