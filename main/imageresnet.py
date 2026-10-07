import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torchvision.transforms import v2
import torchvision.models as models
# from kornia.morphology import erosion, dilation

from . import shards

class ImageResNetShards(shards.Shards):
    class Net(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        def forward(self, x):
            return self.model(x)

    def __init__(self, args, pricing, drift):
        self.classes = 1000
        super().__init__(args, pricing, drift, classes=self.classes)
        if self.args.target == 0:
            self.args.target = 0.85
            print("DEFAULT -- setting target to {}".format(self.args.target))
        # self.norm_mean = 0.1307
        # self.norm_std = 0.3081
        # self.norm_min = -0.42421296
        self.train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                                transforms.RandomHorizontalFlip(0.5),
                                # v2.ElasticTransform(alpha=30.0, sigma=3.0),
                                # v2.RandomPerspective(),
                                # v2.RandomAffine(30, translate=(0.1, 0.1)),
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
        optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=0.175,
                                                  total_steps=50*(1281167//(self.args.bs*self.args.K)),
                                                  epochs=50,
                                                  steps_per_epoch=1281167//(self.args.bs*self.args.K),
                                                  pct_start=0.3,
                                                  div_factor=25.0,
                                                  final_div_factor=1e4)
        # scheduler = optim.lr_scheduler.StepLR(optimizer,
        #                                       step_size=30*(1281167//(self.args.bs*self.args.K)),
        #                                       gamma=0.1)
        # print("total steps per \"epoch\": {}".format(1281167//(self.args.bs)))
        return optimizer, scheduler
    

    # def rand_thicken(self, image:torch.Tensor) -> torch.Tensor:
        # image = torch.unsqueeze(image.float(), 0)
        # t = torch.randint(1, 3, (2,))
        # kernel = torch.ones((t[0], t[1]))
        # if np.random.rand() < 0.5:
        #     return torch.squeeze(erosion(image, kernel=kernel), 0)
        # else:
        #     return torch.squeeze(dilation(image, kernel=kernel), 0)
        
    # def fill_nan(self, image:torch.Tensor) -> torch.Tensor:
    #     return torch.nan_to_num(image, nan=self.norm_min)
