import torch
import torch.nn as nn
from teng_resnet_emo import ResNet_emo, BasicBlock, Bottleneck
from teng_resnet_id import ResNet
from teng_resnet_threed import ResNet_threed,BasicBlock_threed
import torch.utils.model_zoo as model_zoo
from ops.basic_ops import ConsensusModule

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class emo_id_net(nn.Module):
    def __init__(self, num_classes, num_segments, isdropout):
        super(emo_id_net, self).__init__()
        self.num_segments = num_segments
        self.emo = self.resnet18(True)
        self.id = self.idvideomodel(True)
        self.threed = self.resnetthreed(True)
        self.classifier = nn.Sequential(
             nn.Conv2d(512+512,400,1),
         #   nn.Linear(512+512 , num_classes),
        )
        self.idBridge = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x): 
        emo_out,emo_out2 = self.emo(x)      #    [112, 128, 28, 28] [112, 512, 1, 1]
        emo_out = self.dropout(emo_out)
        consensus = ConsensusModule('avg')
        
        emo_out2 = emo_out2.view((-1, self.num_segments) + emo_out2.size()[1:])
        emo_out2 = consensus(emo_out2)
        emo_out2 = emo_out2.squeeze(1)      #     [16, 512, 1, 1]
        emo_out2 = self.dropout(emo_out2)
        
        id_out = self.id(x)                 #     [112, 128, 28, 28]
        id_out = self.dropout(id_out)
        id_out = self.idBridge(id_out)
        idemo_add = torch.add(emo_out,torch.neg(id_out))                            #   [112, 128, 28, 28]
        idemo_add = idemo_add.view((-1, self.num_segments)+ idemo_add.size()[1:])   #   [16, 7, 128, 28, 28]
        idemo_add = torch.transpose(idemo_add, 1, 2)                                #   [16, 128, 7, 28, 28]   
        combine = self.threed(idemo_add)                                            #   [16, 512, 1, 1, 1]
        combine = combine.squeeze(2)              #                                 #   [16, 512, 1, 1]
        bigcat = torch.cat([emo_out2,combine],1)  #                                 #   [16, 1024, 1, 1] 
   #     combine = combine.view(combine.size(0), -1)
   #     emo_out2 = emo_out2.view(-1,512)
   #     bigcat = torch.cat([emo_out2,combine],1)
        bigcat = self.dropout(bigcat)
        bigcat = bigcat.view(-1,1024)                                               #   [16, 1024]
        final = self.classifier(bigcat)
        return final
        
    def resnet18(self,pretrained=False):
        model = ResNet_emo(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        return model
        
    def idvideomodel(self,pretrained=True):
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        model = model.cuda();
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        if pretrained:
            n_finetune_classes = 80
            id_pretrain_path = '/home/developers/tengjianing/myfile/oulu/oulu_vl_s_FD_id_v1_no_ft_rgb_model_best.pth.tar'
            pretrain = torch.load(id_pretrain_path)
            model.load_state_dict(pretrain['state_dict'])
            model.module.fc = nn.Linear(model.module.fc.in_features,n_finetune_classes)
            model.module.fc = model.module.fc.cuda()
        return model
        
    def resnetthreed(self,pretrained=True):
        model = ResNet_threed(BasicBlock_threed, [2, 2, 2, 2], num_classes=400,shortcut_type='A',sample_size=224,sample_duration=16)
        if pretrained:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
            model_dict = model.state_dict()
            new_state_dict = {}
            pretrained_dict_3d = torch.load("/home/developers/tengjianing/myfile/pretrained_model/resnet-18-kinetics.pth")
            for k, v in pretrained_dict_3d['state_dict'].items():
                if (k in model_dict) and (v.size() == model_dict[k].size()):
                    new_state_dict[k] = v
            layer3_0_conv1_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.layer3.0.conv1.weight"], 4, 1)
            new_state_dict["module.layer3.0.conv1.weight"] = torch.cat((layer3_0_conv1_weight_chunk[0], layer3_0_conv1_weight_chunk[1], layer3_0_conv1_weight_chunk[2],layer3_0_conv1_weight_chunk[3]), 1)
            model.load_state_dict(new_state_dict)
        return model
