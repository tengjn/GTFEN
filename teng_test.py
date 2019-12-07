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
import numpy as np
from sklearn.metrics import confusion_matrix
from au import solve_module_problem

id_classes = 80
test_segments = 7
num_crop = 1
num_class = 6
image_tmpl = '{:03d}.jpeg'
device_id = [0]
device = torch.device('cuda:'+str(device_id[0]))

def eval_video(video_data):
    i, data, label = video_data
    input_var = data.view(-1, 3, data.size(2), data.size(3))
    _, output, _, _ = model(input_var)
    rst = output.data.cpu().numpy().copy()
    return i, rst, label[0]
no_partialbn = False
input_mean = [0.5745987,0.49725866,0.46272627]  
input_std = [0.20716324,0.19548155,0.19786908]    
num_classes = 6
num_segments = 7
batch_size = 1

isdropout = True
image_source = '/home/developers/tengjianing/myfile/oulu/video_by_class_frame_vl_s_FD_new_cross_txtsame_id'

i = 3
normalize = GroupNormalize(input_mean, input_std)

val_list = image_source + '/oulu_test_{}.txt'.format(i)
root_path = '/home/developers/tengjianing/myfile/oulu/'

model_root = '/home/developers/tengjianing/another/GTFEN/best_models/same3_2bridge_new/'
model_load_path = model_root + 'fd_minus_18_18_7frames_emo_id_threed_full_fold3_Allse_87.5_model_best.pth.tar'
idpath = "/home/developers/tengjianing/myfile/oulu/seNet18_Oulu_id_rgb_model_best.pth.tar"
id_se_pretrain_path = torch.load(idpath)

test_loader = torch.utils.data.DataLoader(
        TSNDataSet(root_path, val_list, num_segments=num_segments,
                   new_length=1,
                   image_tmpl=image_tmpl,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(244)),
                       GroupCenterCrop(224),
                       Stack(roll=False),
                       ToTensor(norm_value=255),
                       normalize,
                   ])),
        batch_size=1, shuffle=False,
        num_workers=30, pin_memory=True,drop_last=False)


checkpoint = torch.load(model_load_path)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

model = emo_id_net(num_classes,num_segments,id_classes,id_se_pretrain_path).cuda()

model.classifier[0] = nn.Linear(512,6)
model.load_state_dict(solve_module_problem(checkpoint['state_dict']))
print("parameters loading success")
model.to(device)
model = torch.nn.DataParallel(model, device_ids=device_id).cuda()

model.eval()

output = []
total_num = len(test_loader.dataset)
proc_start_time = time.time()
for i, (data, label,_) in enumerate(test_loader):
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
print('total {} video done'.format(total_num))

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
video_labels = [x[1] for x in output]
print(video_labels)
print(video_pred)
cf = confusion_matrix(video_labels, video_pred).astype(float)
print(cf)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
cls_acc = cls_hit / cls_cnt
print(cls_acc)
print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
