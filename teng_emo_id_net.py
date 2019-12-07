import torch
import torch.nn as nn
from backbone.resNet.teng_resnet_emo import ResNet_emo, BasicBlock
from backbone.resNet.teng_resnet_idex import ResNet_idex
from backbone.resNet.teng_resnet_idcl import ResNet_idcl
from backbone.seNet.teng_seNet_idex import SENet_idex
from backbone.seNet.teng_seNet_idcl import SENet_idcl
from backbone.teng_resnet_threed import ResNet_threed,BasicBlock_threed
from backbone.seNet.teng_seNet_emo import Se_BasicBlock, SENet
import torch.utils.model_zoo as model_zoo
from ops.basic_ops import ConsensusModule
from au import solve_module_problem

id_pretrain_path = torch.load('/home/developers/tengjianing/myfile/oulu/oulu_vl_s_FD_id_v1_no_ft_rgb_model_best.pth.tar')
pretrained_dict_3d_path = torch.load("/home/developers/tengjianing/myfile/pretrained_model/resnet-18-kinetics.pth")
se_pretrain_path = torch.load('/home/developers/tengjianing/another/GTFEN/seresnet18.pth')

def se_dict_vary(pretrain_path):
    se_new_dict = {}
    for k in pretrain_path:
        a = k.split('.')
        if 'se_module' in a:
            new_name = a[0] + '.' + a[1] + '.' + a[3] + '.' + a[4]
            if a[4] == 'weight':
                pretrain_path[k] = pretrain_path[k].squeeze(2).squeeze(2)
                se_new_dict[new_name] = pretrain_path[k]
            else:
                se_new_dict[new_name] = pretrain_path[k]
        elif 'layer0' in a:
            new_name = k[7:]
            se_new_dict[new_name] = pretrain_path[k]
        elif 'last_linear' in a:
            new_name = 'fc.' + a[1]
            se_new_dict[new_name] = pretrain_path[k]
        else:
            se_new_dict[k] = pretrain_path[k]
    return se_new_dict
se_new_dict = se_dict_vary(se_pretrain_path)

class emo_id_net(nn.Module):
    def __init__(self, num_classes, num_segments,n_finetune_classes,id_se_pretrain_path):
        super(emo_id_net, self).__init__()
        self.num_segments = num_segments
        self.n_finetune_classes = n_finetune_classes
        self.id_se_pretrain_path = id_se_pretrain_path
        self.emo = self.se_resnet18(True)
        self.idex = self.idextractor_se(True)
        self.threed = self.resnetthreed(True)
        self.classifier = nn.Sequential(
            nn.Linear(512 , num_classes),
        )
        self.idBridge = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.expBridge = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.2)

        
    def forward(self, x): 
        emo_out = self.emo(x)                 #    [112, 128, 28, 28]
        emo_visual = emo_out
        emo_out = self.dropout(emo_out)
        emo_out = self.expBridge(emo_out)
        id_out = self.idex(x)                 #     [112, 128, 28, 28]
        id_visual = id_out
        id_out = self.dropout(id_out)
        #De-coupler
        id_out = self.idBridge(id_out)
        TFE = torch.add(emo_out,torch.neg(id_out))                #   [112, 128, 28, 28]
        TFE_out = TFE                                             #   [16, 7, 128, 28, 28]
        TFE = TFE.view((-1, self.num_segments)+ TFE.size()[1:])   #   [16, 7, 128, 28, 28]
        TFE = torch.transpose(TFE, 1, 2)                          #   [16, 128, 7, 28, 28]
        TFEvector = self.threed(TFE)                              #   [16, 512, 1, 1, 1]
        TFEvector = TFEvector.squeeze(2)                          #   [16, 512, 1, 1]
        TFEvector = self.dropout(TFEvector)
        TFEvector = TFEvector.view(-1,512)                        #   [16, 1024]
        final = self.classifier(TFEvector)
        return TFE_out,final,emo_visual,id_visual
        
    def se_resnet18(self,pretrained=False):
        model = SENet(Se_BasicBlock, [2, 2, 2, 2])
        if pretrained:
            model.load_state_dict(se_new_dict)
        return model

    def idextractor_se(self,pretrained=True):
        model = SENet_idex(Se_BasicBlock, [2, 2, 2, 2], self.n_finetune_classes)
        if pretrained:
            model.load_state_dict(solve_module_problem(self.id_se_pretrain_path['state_dict']))
            model.fc = nn.Linear(model.fc.in_features,self.n_finetune_classes)
        return model

    def resnetthreed(self,pretrained=True):
        model = ResNet_threed(BasicBlock_threed, [2, 2, 2, 2], num_classes=400,shortcut_type='A',sample_size=224,sample_duration=16)
        
        if pretrained:
            model_dict = model.state_dict()
            new_state_dict = {}            
            no_module_state_dict = solve_module_problem(pretrained_dict_3d_path['state_dict'])
            for k, v in no_module_state_dict.items():
                if (k in model_dict) and (v.size() == model_dict[k].size()):
                    new_state_dict[k] = v
            layer3_0_conv1_weight_chunk = torch.chunk(no_module_state_dict["layer3.0.conv1.weight"], 4, 1)
            new_state_dict["layer3.0.conv1.weight"] = torch.cat((layer3_0_conv1_weight_chunk[0], layer3_0_conv1_weight_chunk[1], layer3_0_conv1_weight_chunk[2],layer3_0_conv1_weight_chunk[3]), 1)
            model.load_state_dict(new_state_dict)
        return model

class gan_id_net(nn.Module):
    def __init__(self,num_classes, num_segments,n_finetune_classes,id_se_pretrain_path):
        super(gan_id_net,self).__init__()
        self.num_segments = num_segments
        self.n_finetune_classes = n_finetune_classes
        self.id_se_pretrain_path = id_se_pretrain_path
        self.id_tester = self.idclassifier_se()
        self.classifier = nn.Sequential(nn.Linear(512, self.n_finetune_classes))  #80 for number of id in Oulu

    def forward(self,x):                    #  [112, 128, 28, 28]
        consensus = ConsensusModule('avg')
        id_test = self.id_tester(x)         #  [112, 512, 1, 1]
        id_test = id_test.view((-1, self.num_segments) + id_test.size()[1:])
        id_test = consensus(id_test)
        id_test = id_test.squeeze(1)        #  [16, 512, 1, 1]
        id_test = id_test.view(-1,512)
        id_result = self.classifier(id_test)
        return id_result

    def idclassifier_se(self):
        model = SENet_idcl(Se_BasicBlock, [2, 2, 2, 2], self.n_finetune_classes)
        model_dict = model.state_dict()
        new_state_dict = {}
        no_module_state_dict = solve_module_problem(self.id_se_pretrain_path['state_dict'])
        for k,v in no_module_state_dict.items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        return model