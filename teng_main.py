#!/usr/bin/env python
# coding: utf-8
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.init as init
import os
from torchsummary import summary
import time
import shutil
import torch.nn.parallel
import torch.optim
from dataset_video import TSNDataSet
from teng_emo_id_net import *
from transforms import *
import torch.nn.functional as F
from opts import parse_opts
#from torch.utils.tensorboard import SummaryWriter
import numpy as np

opt=parse_opts()

if opt.dataset == 'oulu':
    input_mean = [0.5745987,0.49725866,0.46272627]
    input_std = [0.20716324,0.19548155,0.19786908]
    num_classes = 6
    image_source = 'video_by_class_frame_vl_s_FD_new_cross_txtsame_id'
    modelDir = 'same3'
    image_tmpl = '{:03d}.jpeg'
    root_path = '/home/developers/tengjianing/myfile/oulu/'
    id_classes = 80
    idpath = "/home/developers/tengjianing/myfile/oulu/seNet18_Oulu_id_rgb_model_best.pth.tar"
    id_se_pretrain_path = torch.load(idpath)
elif opt.dataset == 'ckplus':
    input_mean = [0.48235,0.48235,0.48235]  
    input_std = [0.29019,0.29019,0.29019]    
    num_classes = 7
    image_source = 'ckplus_tenfold_samecroptxt_id'
    modelDir = 'ckplus_same'
    image_tmpl = '{:08d}.png'
    root_path = '/home/developers/tengjianing/myfile/ckplus/'
    id_classes = 118
    idpath = "/home/developers/tengjianing/myfile/ckplus/ckplus_fd_id_v1_rgb_ori_se_resNet18_rgb_model_best.pth.tar"
    id_se_pretrain_path = torch.load(idpath)
else:
    print("invalid dataset input")
    
num_segments = opt.num_segments
epochs = opt.epochs
sstep = opt.sstep
eval_freq = opt.eval_freq
momentum = opt.momentum
lr = opt.lr
weight_decay = opt.weight_decay ###  adjusting
Alpha = opt.Alpha

if not os.path.exists("best_models/" + modelDir):
    os.mkdir("best_models/" + modelDir)

def main():
    print("Freeze_id is: {}".format(opt.freeze_id))
    print("num_segments is: {}".format(num_segments))
    print("weight_decay is: {}".format(weight_decay))
    print("image_source is {}".format(image_source))
    print("num_classes is {}".format(num_classes))
    print("id classes is {}".format(id_classes))
    print("id_pretrain path is {}".format(idpath))
    print("Alpha is {}".format(Alpha))
    global best_prec1
    global snapshot_pref
#    writer = SummaryWriter()
    final_accuracy = 0
    ten_fold_best = []
    for i in range(10):    
        if i == 0 or i == 5 or i == 6 or i== 7 or i == 9:
            continue
        best_prec1 = 0
        normalize = GroupNormalize(input_mean, input_std)
        snapshot_pref = "best_models/" + modelDir + "/fd_minus_18_18_7frames_emo_id_threed_full_fold{}_Allse".format(i)
        train_augmentation = torchvision.transforms.Compose([GroupScale(244),
                                                            GroupRandomCrop(opt.input_size),
                                                           GroupRandomHorizontalFlip(is_flow=False)])
        if opt.dataset == 'oulu':
            train_list = image_source + '/oulu_train_{}.txt'.format(i)
            val_list = image_source + '/oulu_test_{}.txt'.format(i)
        elif opt.dataset == 'ckplus':
            train_list = image_source + '/ckplus_train_{}.txt'.format(i)
            val_list = image_source + '/ckplus_test_{}.txt'.format(i)        
        print(train_list)
        print(val_list)
    
        train_loader = torch.utils.data.DataLoader(
                dataset = TSNDataSet(root_path, train_list, num_segments=num_segments,
                             new_length=1,
                             image_tmpl=image_tmpl,
                             transform=torchvision.transforms.Compose([
                             train_augmentation,
                             Stack(roll=False),
                             ToTorchFormatTensor(div=True),
                             normalize,
                             ])),
                batch_size=opt.batch_size, shuffle=True,
                num_workers=30, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
                TSNDataSet(root_path, val_list, num_segments=num_segments,
                           new_length=1,
                           image_tmpl=image_tmpl,
                           random_shift=False,
                           transform=torchvision.transforms.Compose([
                               GroupScale(int(244)),
                               GroupCenterCrop(opt.input_size),
                               Stack(roll=False),
                               ToTorchFormatTensor(div=True),
                               normalize,
                           ])),
                batch_size=opt.batch_size, shuffle=False,
                num_workers=30, pin_memory=True,drop_last=False)
    
        model_G = emo_id_net(num_classes,num_segments,id_classes,id_se_pretrain_path).cuda()
        model_D = gan_id_net(num_classes,num_segments,id_classes,id_se_pretrain_path).cuda()
        if i == 0:
            summary(model_G, (21, 224, 224))
        initNetParams_G(model_G)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        criterion_MSE = torch.nn.MSELoss().cuda()
        optimizer_G = torch.optim.SGD(model_G.parameters(), lr, momentum = momentum, weight_decay = weight_decay)
        optimizer_D = torch.optim.SGD(model_D.parameters(), lr, momentum = momentum, weight_decay = weight_decay)

        for epoch in range(0, epochs):
                adjust_learning_rate(optimizer_G,optimizer_D, epoch, lr)
                if epoch % 3 == 0:
                     train_D(train_loader, model_G, model_D, criterion, criterion_MSE, optimizer_D, epoch)
                else:
                  train_G(train_loader, model_G, model_D, criterion, criterion_MSE, optimizer_G, epoch)
                if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
                    prec1 = validate(val_loader, model_G, criterion,(epoch + 1) * len(train_loader))

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model_G.state_dict(),
                        'best_prec1': best_prec1,
                    }, is_best, best_prec1)
        ten_fold_best.append(best_prec1)
        final_accuracy = final_accuracy + best_prec1
    print(ten_fold_best)
    print("Final accuracy is {}".format(final_accuracy))

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

def train_G(train_loader, model_G,model_D, criterion, criterion_MSE, optimizer_G, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()
    model_G.train()
    model_D.train()

    if opt.freeze_id:
        for param in model_G.idex.parameters():
            param.requires_grad = False
        
    for i, (input, target, target_id) in enumerate(train_loader):
        id_one_hot = F.one_hot(target_id, num_classes=id_classes).cuda().cuda()
        target = target.cuda()
        target_id = target.cuda()
        TFE_out, output, _, _ = model_G(input)
        output_id = model_D(TFE_out)

        loss_exp = criterion(output, target)
        loss_id = criterion_MSE(output_id, id_one_hot.float())
   #     loss_id = criterion(output_id, target_id_var)
        loss = Alpha * loss_exp - loss_id

        prec1,_ = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        if i % 10 == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), loss=losses, top1=top1, lr=optimizer_G.param_groups[-1]['lr']))
            print(output)

def train_D(train_loader, model_G, model_D, criterion,criterion_MSE, optimizer_D, epoch):

    model_G.train()
    model_D.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    for i, (input, target, target_id) in enumerate(train_loader):
        target_id = target_id.cuda()
        TFE_out, _, _, _ = model_G(input)
        output_id = model_D(TFE_out)
        loss = criterion(output_id, target_id)

        prec1,_ = accuracy(output_id.data, target_id, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()

        if i % 10 == 0:
            output = ('NetD Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), loss=losses, top1=top1))
            print(output)

def validate(val_loader, model_G, criterion,iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model_G.eval()

    for i, (input, target, target_id) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        _,output, _, _ = model_G(input_var)
        loss = criterion(output, target_var)
        prec1,_ = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time

        if i % 10 == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, loss=losses))
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

def initNetParams_G(model):
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
    for m in model.idBridge.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def adjust_learning_rate(optimizer_G,optimizer_D, epoch, lr):
    lr = lr * (0.1 ** (epoch // sstep))
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = lr
    
if __name__ == '__main__':
    main()