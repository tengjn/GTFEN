import torch
import torch.nn as nn
from teng_resnet_emo import ResNet_emo, BasicBlock
from teng_resnet_idex import ResNet_idex
from teng_resnet_idcl import ResNet_idcl
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
id_pretrain_path = torch.load('/home/developers/tengjianing/myfile/oulu/oulu_vl_s_FD_id_v1_no_ft_rgb_model_best.pth.tar')
pretrained_dict_3d_path = torch.load("/home/developers/tengjianing/myfile/pretrained_model/resnet-18-kinetics.pth")

class emo_id_net(nn.Module):
    def __init__(self, num_classes, num_segments):
        super(emo_id_net, self).__init__()
        self.num_segments = num_segments
        self.emo = self.resnet18(True)
        self.idex = self.idextractor(True)
        self.threed = self.resnetthreed(True)
        self.classifier = nn.Sequential(
            nn.Linear(512 , num_classes),
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
        emo_out = self.emo(x)      #    [112, 128, 28, 28] [112, 512, 1, 1]
        emo_out = self.dropout(emo_out)

        id_out = self.idex(x)                 #     [112, 128, 28, 28]
        id_out = self.dropout(id_out)
        #De-coupler
        id_out = self.idBridge(id_out)
        TFE = torch.add(emo_out,torch.neg(id_out))                            #   [112, 128, 28, 28]
        TFE_out = TFE
        TFE = TFE.view((-1, self.num_segments)+ TFE.size()[1:])   #   [16, 7, 128, 28, 28]
        TFE = torch.transpose(TFE, 1, 2)                                #   [16, 128, 7, 28, 28]
        TFEvector = self.threed(TFE)                                            #   [16, 512, 1, 1, 1]
        TFEvector = TFEvector.squeeze(2)              #                                 #   [16, 512, 1, 1]
        TFEvector = self.dropout(TFEvector)
        TFEvector = TFEvector.view(-1,512)                                               #   [16, 1024]\\
        final = self.classifier(TFEvector)
        return TFE_out,final
        
    def resnet18(self,pretrained=False):
        model = ResNet_emo(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        return model
        
    def idextractor(self,pretrained=True):
        model = ResNet_idex(BasicBlock, [2, 2, 2, 2])
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        if pretrained:
            n_finetune_classes = 80
            model.load_state_dict(id_pretrain_path['state_dict'])
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
            for k, v in pretrained_dict_3d_path['state_dict'].items():
                if (k in model_dict) and (v.size() == model_dict[k].size()):
                    new_state_dict[k] = v
            layer3_0_conv1_weight_chunk = torch.chunk(pretrained_dict_3d_path["state_dict"]["module.layer3.0.conv1.weight"], 4, 1)
            new_state_dict["module.layer3.0.conv1.weight"] = torch.cat((layer3_0_conv1_weight_chunk[0], layer3_0_conv1_weight_chunk[1], layer3_0_conv1_weight_chunk[2],layer3_0_conv1_weight_chunk[3]), 1)
            model.load_state_dict(new_state_dict)
        return model


class gan_id_net(nn.Module):
    def __init__(self,num_classes, num_segments):
        super(gan_id_net,self).__init__()
        self.num_segments = num_segments
        self.id_tester = self.idclassifier()
        self.classifier = nn.Sequential(nn.Linear(512 , 80))  #80 for number of id in Oulu

    def forward(self,x):                    #  [112, 128, 28, 28]
        consensus = ConsensusModule('avg')
        id_test = self.id_tester(x)         #  [112, 512, 1, 1]
        id_test = id_test.view((-1, self.num_segments) + id_test.size()[1:])
        id_test = consensus(id_test)
        id_test = id_test.squeeze(1)        #  [16, 512, 1, 1]
        id_test = id_test.view(-1,512)
        id_result = self.classifier(id_test)
        return id_result

    def idclassifier(self):
        model = ResNet_idcl(BasicBlock, [2, 2, 2, 2])
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        model_dict = model.state_dict()
        new_state_dict = {}
        for k,v in id_pretrain_path['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        return model



