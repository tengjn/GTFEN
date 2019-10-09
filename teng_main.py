#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.init as init
import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.basic_ops import ConsensusModule
from dataset_video import TSNDataSet
from teng_emo_id_net import *
from transforms import *

test_mode = False
isPretrainedOnKinetic = True
freeze_id = True
consensus_type = 'avg'
consensus = ConsensusModule(consensus_type)
no_partialbn = False
input_mean = [0.5745987,0.49725866,0.46272627]  
input_std = [0.20716324,0.19548155,0.19786908]       
num_classes = 6
num_segments = 7
batch_size = 16
epochs = 200
sstep = 70
eval_freq = 5

clip_gradient = 20
momentum = 0.9
input_size = 224
crop_size = input_size
scale_size = 224
lr = 0.001
isdropout = True
rotate_DA = 5
bright_DA = None
weight_decay = 0.0001 ###  adjusting
image_source = '/home/developers/tengjianing/myfile/oulu/video_by_class_frame_vl_s_FD_new_cross_txtsame'
Log_name = "same3"
if not os.path.exists("best_models/" + Log_name):
    os.mkdir("best_models/" + Log_name)

def main():
    print("Freeze_id is: {}".format(freeze_id))
    print("num_segments is: {}".format(num_segments))
    print("weight_decay is: {}".format(weight_decay))
    print("rotate_DA is {}".format(rotate_DA))
    print("bright_DA is {}".format(bright_DA))
    print("isPretrainedOnKinetic is {}".format(isPretrainedOnKinetic))
    print("image_source is {}".format(image_source))
    global best_prec1
    global snapshot_pref
    ten_fold_best = []
    for i in range(10):    
        best_prec1 = 0
        normalize = GroupNormalize(input_mean, input_std)
        snapshot_pref = "best_models/" + Log_name + "/oulu_fd_minus_18_18_7frames_emo_id_threed_full_fold{}_pretrained".format(i)
        train_augmentation = torchvision.transforms.Compose([GroupScale(244),
                                                            GroupRandomCrop(224),
                                                           GroupRandomHorizontalFlip(is_flow=False)])
    
        train_list = image_source + '/oulu_train_{}.txt'.format(i)
        val_list = image_source + '/oulu_test_{}.txt'.format(i)
        print(train_list)
        print(val_list)
        root_path = '/home/developers/tengjianing/myfile/oulu/'
    
        train_loader = torch.utils.data.DataLoader(
                dataset = TSNDataSet(root_path, train_list, num_segments=num_segments,
                             new_length=1,
                             image_tmpl='{:03d}.jpeg',
                             transform=torchvision.transforms.Compose([
                             train_augmentation,
                             Stack(roll=False),
                             ToTorchFormatTensor(div=True),
                             normalize,
                             ])),
                batch_size=batch_size, shuffle=True,
                num_workers=30, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
                TSNDataSet(root_path, val_list, num_segments=num_segments,
                           new_length=1,
                           image_tmpl='{:03d}.jpeg',
                           random_shift=False,
                           transform=torchvision.transforms.Compose([
                               GroupScale(int(244)),
                               GroupCenterCrop(224),
                               Stack(roll=False),
                               ToTorchFormatTensor(div=True),
                               normalize,
                           ])),
                batch_size=batch_size, shuffle=False,
                num_workers=30, pin_memory=True,drop_last=False)
    
        model = emo_id_net(num_classes,num_segments,isdropout)
        if(isPretrainedOnKinetic):
            pretrain_path = '/home/developers/tengjianing/myfile/tsn-pytorch-minus-threed-full/oulu_fd_minus_18_18_7frames_emo_id_threed_full_pretrained_kinectic_rgb_model_best.pth.tar'
            pretrain = torch.load(pretrain_path)
            model.load_state_dict(pretrain['state_dict'])
            model.classifier[0] = nn.Linear(512+512,6)
        model = model.cuda()
        initNetParams(model)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(),lr,momentum = momentum,weight_decay = weight_decay)
        for epoch in range(0, epochs):
                adjust_learning_rate(optimizer, epoch, lr)
                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch)
    
                if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
                    prec1 = validate(val_loader, model, criterion,(epoch + 1) * len(train_loader))
    
                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    }, is_best, best_prec1)
        ten_fold_best.append(best_prec1)
    print(ten_fold_best)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    
    if freeze_id:
        for param in model.id.parameters():
            param.requires_grad = False
        
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)

        loss = criterion(output, target_var)       

        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        if clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), clip_gradient)
        optimizer.step()

        if i % 10 == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)

def validate(val_loader, model, criterion,iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
               
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.item(), input.size(0))                  # pytorch 0.4 version
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # measure elapsed time

        if i % 10 == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    return top1.avg

def save_checkpoint(state, is_best, best_prec1, filename='checkpoint.pth.tar'):
    filename = '_'.join((snapshot_pref,  filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((snapshot_pref, str(best_prec1), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def initNetParams(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
  #      elif isinstance(m, nn.Conv3d):
  #         init.xavier_normal(m.weight)
  #      elif isinstance(m, nn.BatchNorm3d):
  #          init.constant(m.weight, 1)
  #          init.constant(m.bias, 0)
def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // sstep))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    
if __name__ == '__main__':
    main()