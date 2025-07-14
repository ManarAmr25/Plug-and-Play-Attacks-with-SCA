import torch
import torchvision
from tqdm import tqdm
from torch import nn, optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as T
import pandas as pd
import math
import numpy as np
import os
import torch.nn.functional as F
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from lcapt.lca import LCAConv2D
import heapq
import time
import random as rand
from numpy.random import random
from PIL import Image
import PIL.Image
# import dnnlib
# import legacy
import fnmatch
import sys
import pickle
from torchvision.models import densenet, inception, resnet

class LinearNoDefSplitNN(nn.Module):
  def __init__(self, num_classes=10):
    super(LinearNoDefSplitNN, self).__init__()

    self.num_classes = num_classes

    self.first_part = nn.Sequential(
                           nn.Linear(28, 500),
                           nn.ReLU(),
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(500, 500),
                           nn.ReLU(),
                           nn.Linear(500, 28),
                           nn.ReLU(), 
                           nn.Linear(28, 500),                        )
    self.third_part = nn.Sequential(
                           nn.Linear(1*28*500, self.num_classes),
                           #scancel nn.Softmax(dim=-1),
                         )

  def forward(self, x):
     #x=x.view(-1,32*32*3)
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    #print(x.shape)
    x=self.second_part(x)
    #print(x.shape)
    x = x.view(-1, 1*28*500)
    x=self.third_part(x)
    return x
  
class LinearScaSplitNN(nn.Module):
  def __init__(self, num_classes=10):
    super(LinearScaSplitNN, self).__init__()

    self.num_classes = num_classes

    self.first_part = nn.Sequential(
        LCAConv2D(out_neurons=16,
                in_neurons=1,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.5, lca_iters=500, pad="same",            
            ),   
            #nn.BatchNorm2d(16),                           
       LCAConv2D(out_neurons=28,
                in_neurons=16,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.5, lca_iters=500, pad="same",            
            ),                           nn.Linear(28, 500),
                           nn.ReLU(),
 
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(500, 500),
                           nn.ReLU(),
                           nn.Linear(500, 28),
                           nn.ReLU(), 
                           nn.Linear(28, 500),                        )
    self.third_part = nn.Sequential(
                           nn.Linear(28*28*500, self.num_classes),
                           #scancel nn.Softmax(dim=-1),
                         )

  def forward(self, x):
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    #print(x.shape)
    x=self.second_part(x)
    x = x.view(-1, 28*28*500)
    x=self.third_part(x)
    return x

class NoDefSplitNN(nn.Module):
  def __init__(self, num_classes=10):
    super(NoDefSplitNN, self).__init__()

    self.num_classes = num_classes

    self.first_part = nn.Sequential(
       nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(), 
            nn.Dropout(.09), 
            #nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(16) ,                   
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(), 
            nn.Dropout(.09), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),                    

                         )
    self.second_part = nn.Sequential(
                           nn.Conv2d(32, 64, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(64),   
                            nn.Conv2d(64, 128, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(128), 
                            nn.Conv2d(128, 256, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09),  
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(256),  
                            
                           #scancel nn.Softmax(dim=-1),
                         )
    self.third_part = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            # nn.Linear(256 * 14 * 14, 512),
                            nn.ReLU(),
                            # nn.Linear(512, 5),
                            nn.Linear(512, num_classes)
    )

  def forward(self, x):
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    #x = x.view(-1, 32*16*500)
    #print(x.shape)
    x=self.second_part(x)
    # print(">>>> before view: ", x.shape)
    x = x.view(-1, 256*2*2)
    # x = x.view(x.shape[0], -1)
    # print(">>>> after view: ", x.shape)
    x=self.third_part(x)
    # print(">>>> after 3rd part: ", x.shape)

    return x

class ScaSplitNN(nn.Module):
  def __init__(self, num_classes=10):
    super(ScaSplitNN, self).__init__()
    print('...... created ScaSplitNN')

    self.num_classes = num_classes

    self.first_part = nn.Sequential(
            LCAConv2D(out_neurons=16,
                in_neurons=3,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.1, lca_iters=500, pad="same",               
            ),                               
            nn.ReLU(), 
            nn.Dropout(.09), 
            #nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(16) ,                   
            LCAConv2D(out_neurons=32,
                in_neurons=16,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.1, lca_iters=500, pad="same", ),    
            nn.ReLU(), 
            nn.Dropout(.09), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),                    

                         )
    self.second_part = nn.Sequential(
                           nn.Conv2d(32, 64, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(64),   
                            nn.Conv2d(64, 128, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09), 
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(128), 
                            nn.Conv2d(128, 256, 5, 1, 2),     
                            nn.ReLU(), 
                            nn.Dropout(.09),  
                            nn.MaxPool2d(2, 2),
                            nn.BatchNorm2d(256),  
                            
                           #scancel nn.Softmax(dim=-1),
                         )
    self.third_part = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            # nn.Linear(256 * 14 * 14, 512),
                            nn.ReLU(),
                            # nn.Linear(512, 10),  
                            nn.Linear(512, num_classes)

    )

  def forward(self, x):
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    #x = x.view(-1, 32*16*500)
    #print(x.shape)
    x=self.second_part(x)
    #print(x.shape)
    x = x.view(-1, 256*2*2)
    # x = x.view(x.shape[0], -1)
    x=self.third_part(x)

    return x
  
  
class SCAResNet152(nn.Module):
    def __init__(
        self, pretrained=True, num_classes=10
    ) -> None:
        super().__init__()
        model = resnet.resnet152(pretrained=True)
        self.first_part = nn.Sequential(
            LCAConv2D(out_neurons=16,
                        in_neurons=3,                        
                        kernel_size=5,              
                        stride=1,     
                        tau=1000,               
                        lambda_=0.5, lca_iters=500, pad="same",               
                    ),                               
                    nn.ReLU(), 
                    nn.Dropout(.09), 
                    #nn.MaxPool2d(2, 2), 
                    nn.BatchNorm2d(16) ,                   
            LCAConv2D(out_neurons=32,
                        in_neurons=16,                        
                        kernel_size=5,              
                        stride=1, 
                        tau=1000,                   
                        lambda_=0.5, lca_iters=500, pad="same", ),    
                                                nn.ReLU(), 
                    nn.Dropout(.09), 
                    # nn.MaxPool2d(2, 2), # don't downsample from 32 --> 16
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 3, kernel_size=1)   # adapter                

                                )
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            nn.Conv2d(2048, 1024, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(1024, 512, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.fc = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_classes),
            )

    def first_part_forward(self,x):
        x = self.first_part(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.downsample(x)
        
        return x
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.first_part_forward(x)
        # print("before classifier",x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
      
class SCAResNet18(nn.Module):
    def __init__(
        self, pretrained=True, num_classes=10
    ) -> None:
        super().__init__()
        model = resnet.resnet18(pretrained=True)
        self.first_part = nn.Sequential(
            LCAConv2D(out_neurons=16,
                        in_neurons=3,                        
                        kernel_size=5,              
                        stride=1,     
                        tau=1000,               
                        lambda_=0.5, lca_iters=500, pad="same",               
                    ),                               
                    nn.ReLU(), 
                    nn.Dropout(.09), 
                    #nn.MaxPool2d(2, 2), 
                    nn.BatchNorm2d(16) ,                   
            LCAConv2D(out_neurons=32,
                        in_neurons=16,                        
                        kernel_size=5,              
                        stride=1, 
                        tau=1000,                   
                        lambda_=0.5, lca_iters=500, pad="same", ),    
                                                nn.ReLU(), 
                    nn.Dropout(.09), 
                    # nn.MaxPool2d(2, 2), # don't downsample from 32 --> 16
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 3, kernel_size=1)   # adapter                

                                )
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            # nn.Conv2d(2048, 1024, kernel_size=1),  # Reduce channels: 512 → 256
            # nn.Conv2d(1024, 512, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.fc = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_classes),
            )

    def first_part_forward(self,x):
        x = self.first_part(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.downsample(x)
        
        return x
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.first_part_forward(x)
        # print("before classifier",x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
      
class SCAVGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super().__init__()
        # base = models.vgg16(pretrained=pretrained)
        model = torchvision.models.vgg16_bn(pretrained=True)

        self.first_part = nn.Sequential(
            LCAConv2D(out_neurons=16,
                        in_neurons=3,                        
                        kernel_size=5,              
                        stride=1,     
                        tau=1000,               
                        lambda_=0.5, lca_iters=500, pad="same",               
                    ),                               
                    nn.ReLU(), 
                    nn.Dropout(.09), 
                    #nn.MaxPool2d(2, 2), 
                    nn.BatchNorm2d(16) ,                   
            LCAConv2D(out_neurons=32,
                        in_neurons=16,                        
                        kernel_size=5,              
                        stride=1, 
                        tau=1000,                   
                        lambda_=0.5, lca_iters=500, pad="same", ),    
                                                nn.ReLU(), 
                    nn.Dropout(.09), 
                    # nn.MaxPool2d(2, 2), # don't downsample from 32 --> 16
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 3, kernel_size=1)   # adapter                

                                )
        # Use pretrained convolutional layers
        self.features = model.features  # conv blocks
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.classifier = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_classes),
            )
        print("classifier -->",self.classifier)
    def forward(self, x):
        x = self.first_part(x)
        x = self.features(x)
        x = self.downsample(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
      
class ResNet152(nn.Module):
    def __init__(
        self, pretrained=True, num_classes=10
    ) -> None:
        super().__init__()
        model = resnet.resnet152(pretrained=True)
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            nn.Conv2d(2048, 1024, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(1024, 512, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.fc = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_classes),
            )

    def first_part_forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.downsample(x)
        
        return x
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.first_part_forward(x)
        # print("before classifier",x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
      
class ResNet18(nn.Module):
    def __init__(
        self, pretrained=True, num_classes=10
    ) -> None:
        super().__init__()
        model = resnet.resnet18(pretrained=True)
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            # nn.Conv2d(2048, 1024, kernel_size=1),  # Reduce channels: 512 → 256
            # nn.Conv2d(1024, 512, kernel_size=1),  # Reduce channels: 512 → 256
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.fc = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_classes),
            )

    def first_part_forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.downsample(x)
        
        return x
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.first_part_forward(x)
        # print("before classifier",x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
      
      
class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super().__init__()
        
        model = torchvision.models.vgg16_bn(pretrained=True)

        # Use pretrained convolutional layers
        self.features = model.features  # conv blocks
        # self.avgpool = model.avgpool

        # Insert SCL between features and classifier
        # self.scl = SparseCodingLayer(in_channels=512, out_channels=512)

        # Use the original classifier (or modify for custom num_classes)
        self.downsample = nn.Sequential(
            # model.avgpool,
            nn.AdaptiveAvgPool2d((2, 2)),     # [16, 512, 7, 7] -> [16, 512, 2, 2]
            nn.Conv2d(512, 256, kernel_size=1)  # Reduce channels: 512 → 256
        )
        self.classifier = nn.Sequential(
                            nn.Linear(256*2*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        # x = self.scl(x)  # sparse coding layer
        # print("shape before downsample:",x.shape)
        # print("shape of adaptive pooling 2",nn.AdaptiveAvgPool2d((2, 2))(x).shape)
        x = self.downsample(x)
        # print("After downsample",x.shape)
        # print("*****")
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
      
      
