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
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            self.features = torch.nn.Sequential(*list(model.children()))

        def train(self, mode=True):
            """
            Override the default train() to freeze the BN parameters
            """
            super().train(mode)
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        def forward(self, x):
            # return self.model(x)
            return self.features(x)

    # class Net(nn.Module):
    #     class Bottleneck(nn.Module):

    #         def __init__(self,in_channels,intermediate_channels,expansion,is_Bottleneck,stride):
                
    #             """
    #             Creates a Bottleneck with conv 1x1->3x3->1x1 layers.
                
    #             Note:
    #             1. Addition of feature maps occur at just before the final ReLU with the input feature maps
    #             2. if input size is different from output, select projected mapping or else identity mapping.
    #             3. if is_Bottleneck=False (3x3->3x3) are used else (1x1->3x3->1x1). Bottleneck is required for resnet-50/101/152
    #             Args:
    #                 in_channels (int) : input channels to the Bottleneck
    #                 intermediate_channels (int) : number of channels to 3x3 conv 
    #                 expansion (int) : factor by which the input #channels are increased
    #                 stride (int) : stride applied in the 3x3 conv. 2 for first Bottleneck of the block and 1 for remaining

    #             Attributes:
    #                 Layer consisting of conv->batchnorm->relu

    #             """

    #             super().__init__()

    #             self.expansion = expansion
    #             self.in_channels = in_channels
    #             self.intermediate_channels = intermediate_channels
    #             self.is_Bottleneck = is_Bottleneck
                
    #             # i.e. if dim(x) == dim(F) => Identity function
    #             if self.in_channels==self.intermediate_channels*self.expansion:
    #                 self.identity = True
    #             else:
    #                 self.identity = False
    #                 projection_layer = []
    #                 projection_layer.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels*self.expansion, kernel_size=1, stride=stride, padding=0, bias=False ))
    #                 projection_layer.append(nn.BatchNorm2d(self.intermediate_channels*self.expansion))
    #                 # Only conv->BN and no ReLU
    #                 # projection_layer.append(nn.ReLU())
    #                 self.projection = nn.Sequential(*projection_layer)

    #             # commonly used relu
    #             self.relu = nn.ReLU()

    #             # is_Bottleneck = True for all ResNet 50+
    #             if self.is_Bottleneck:
    #                 # bottleneck
    #                 # 1x1
    #                 self.conv1_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False )
    #                 self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)
                    
    #                 # 3x3
    #                 self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False )
    #                 self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)
                    
    #                 # 1x1
    #                 self.conv3_1x1 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False )
    #                 self.batchnorm3 = nn.BatchNorm2d( self.intermediate_channels*self.expansion )
                
    #             else:
    #                 # basicblock
    #                 # 3x3
    #                 self.conv1_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False )
    #                 self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)
                    
    #                 # 3x3
    #                 self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False )
    #                 self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)

    #         def forward(self,x):
    #             # input stored to be added before the final relu
    #             in_x = x

    #             if self.is_Bottleneck:
    #                 # conv1x1->BN->relu
    #                 x = self.relu(self.batchnorm1(self.conv1_1x1(x)))
                    
    #                 # conv3x3->BN->relu
    #                 x = self.relu(self.batchnorm2(self.conv2_3x3(x)))
                    
    #                 # conv1x1->BN
    #                 x = self.batchnorm3(self.conv3_1x1(x))
                
    #             else:
    #                 # conv3x3->BN->relu
    #                 x = self.relu(self.batchnorm1(self.conv1_3x3(x)))

    #                 # conv3x3->BN
    #                 x = self.batchnorm2(self.conv2_3x3(x))


    #             # identity or projected mapping
    #             if self.identity:
    #                 x += in_x
    #             else:
    #                 x += self.projection(in_x)

    #             # final relu
    #             x = self.relu(x)
                
    #             return x

    #     def __init__(self, num_classes, resnet_variant=([64,128,256,512],[3,4,6,3],4,True), in_channels=3):
    #         """
    #         Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
    #         Based on the input parameters, define the channels list, repeatition list along with expansion factor(4) and stride(3/1)
    #         using _make_blocks method, create a sequence of multiple Bottlenecks
    #         Average Pool at the end before the FC layer 

    #         Args:
    #             resnet_variant (list) : eg. [[64,128,256,512],[3,4,6,3],4,True]
    #             in_channels (int) : image channels (3)
    #             num_classes (int) : output #classes 

    #         Attributes:
    #             Layer consisting of conv->batchnorm->relu

    #         """
    #         super().__init__()
    #         self.channels_list = resnet_variant[0]
    #         self.repeatition_list = resnet_variant[1]
    #         self.expansion = resnet_variant[2]
    #         self.is_Bottleneck = resnet_variant[3]

    #         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False )
    #         self.batchnorm1 = nn.BatchNorm2d(64)
    #         self.relu = nn.ReLU()

    #         self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    #         self.block1 = self._make_blocks( 64 , self.channels_list[0], self.repeatition_list[0], self.expansion, self.is_Bottleneck, stride=1 )
    #         self.block2 = self._make_blocks( self.channels_list[0]*self.expansion , self.channels_list[1], self.repeatition_list[1], self.expansion, self.is_Bottleneck, stride=2 )
    #         self.block3 = self._make_blocks( self.channels_list[1]*self.expansion , self.channels_list[2], self.repeatition_list[2], self.expansion, self.is_Bottleneck, stride=2 )
    #         self.block4 = self._make_blocks( self.channels_list[2]*self.expansion , self.channels_list[3], self.repeatition_list[3], self.expansion, self.is_Bottleneck, stride=2 )

    #         self.average_pool = nn.AdaptiveAvgPool2d(1)
    #         self.fc1 = nn.Linear( self.channels_list[3]*self.expansion , num_classes)

    #     # def train(self, mode=True):
    #     #     """
    #     #     Override the default train() to freeze the BN parameters
    #     #     """
    #     #     super().train(mode)

    #     #     # print("Freezing Mean/Var of BatchNorm2D.")
    #     #     # print("Freezing Weight/Bias of BatchNorm2D.")

    #     #     for m in self.features.modules():
    #     #         if isinstance(m, nn.BatchNorm2d):
    #     #             m.eval()
    #     #             m.weight.requires_grad = False
    #     #             m.bias.requires_grad = False

    #     def forward(self,x):
    #         x = self.relu(self.batchnorm1(self.conv1(x)))
    #         x = self.maxpool(x)
            
    #         x = self.block1(x)
            
    #         x = self.block2(x)
            
    #         x = self.block3(x)
            
    #         x = self.block4(x)
            
    #         x = self.average_pool(x)

    #         x = torch.flatten(x, start_dim=1)
    #         x = self.fc1(x)
            
    #         return x

    #     def _make_blocks(self,in_channels,intermediate_channels,num_repeat, expansion, is_Bottleneck, stride):
            
            # """
            # Args:
            #     in_channels : #channels of the Bottleneck input
            #     intermediate_channels : #channels of the 3x3 in the Bottleneck
            #     num_repeat : #Bottlenecks in the block
            #     expansion : factor by which intermediate_channels are multiplied to create the output channels
            #     is_Bottleneck : status if Bottleneck in required
            #     stride : stride to be used in the first Bottleneck conv 3x3

            # Attributes:
            #     Sequence of Bottleneck layers

            # """
            # layers = [] 

            # layers.append(self.Bottleneck(in_channels,intermediate_channels,expansion,is_Bottleneck,stride=stride))
            # for num in range(1,num_repeat):
            #     layers.append(self.Bottleneck(intermediate_channels*expansion,intermediate_channels,expansion,is_Bottleneck,stride=1))

            # return nn.Sequential(*layers)

    def __init__(self, args, pricing, drift):
        self.classes = 1000
        super().__init__(args, pricing, drift, classes=self.classes)
        if self.args.target == 0:
            self.args.target = 0.60
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
                                                  pct_start=0.3,
                                                  div_factor=25.0,
                                                  final_div_factor=1e-4)
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
