#!/usr/bin/env python
# coding: utf-8

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo

traindir = '/home/developers/tengjianing/myfile/oulu/video_by_class_frame_vl_s_FD_id_cross/id_train/'
valdir = '/home/developers/tengjianing/myfile/oulu/video_by_class_frame_vl_s_FD_id_cross/id_train/'

train_loader = data.DataLoader(
    datasets.ImageFolder(traindir,
                         transforms.Compose([
                             transforms.Scale(224),
                             transforms.ToTensor(),]
                         )),
    batch_size=32,
    shuffle=True,
    num_workers=1,
    pin_memory=False)

def train(train_loader, epoch):
    
    for i, (input, target) in enumerate(train_loader):
        print('aaa')

for epoch in range(0, 400):
    train(train_loader, epoch)