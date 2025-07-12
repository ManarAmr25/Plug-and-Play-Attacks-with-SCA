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