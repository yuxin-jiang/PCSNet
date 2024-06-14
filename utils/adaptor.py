import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from utils.coordconv import CoordConv2d
import torch.nn.functional as F
ROTATION_OBJECTS = ['grid', 'screw','bracket_black', 'bracket_brown', 'tubes']

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class adaptor(nn.Module):
    def __init__(self, model, data_loader, gamma_c, device, class_name):
        super(adaptor, self).__init__()
        self.device = device
        self.class_name = class_name
        self.C = 0
        self.nu = 1e-3
        self.scale = None
        self.gamma_c = gamma_c
        self.alpha = 1e-1*2
        self.K = 3
        self.r = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.Descriptor = Descriptor(self.class_name).to(device)
        self._init_centroid(model, data_loader)
        self.C = rearrange(self.C, 'b c h w -> (b h w) c').detach()
        self.C = self.C.transpose(-1, -2).detach()
        self.C = nn.Parameter(self.C, requires_grad=False)
        self.topk_rate = 0.1

    def forward(self, p, label, mask):

        PHI_P, p = self.Descriptor(p)
        phi_p = rearrange(PHI_P, 'b c h w -> b (h w) c')
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(phi_p, (self.C))
        dist = features + centers - f_c
        dist = torch.sqrt(dist)
        n_neighbors = 200
        score = dist.topk(n_neighbors, largest=False).values
        score = rearrange(score, 'b (h w) c -> b c h w', h=self.scale)
        if self.training:
            loss, score_1 = self._soft_boundary(dist, label, mask)
            return loss, score_1, score,PHI_P[:, :896, :, :]
        return score, PHI_P[:, :896, :, :]

    def _soft_boundary(self,dist, label, mask):
        ############################################
        n_neighbors = 3
        if label == 0:## normal feature
            dist = dist.topk(n_neighbors, largest=False).values
            score_1 = dist
            score = (dist[:, :, :] - self.r ** 2)
            L_att = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
            loss = L_att

        if label == 1:## abnormal feature
            dist0 = rearrange(dist, 'b (h w) c -> b c h w', h=self.scale)
            dist0 = dist0 * mask
            dist = dist0.topk(n_neighbors, largest=False).values
            dist = ((self.r+self.alpha) ** 2 - dist[:, :, :])
            score_1 = dist
            L_rep = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score_1), score_1))
            loss =  L_rep

        if label == 2:## Some parts are normal, some parts are abnormal.
            dist = dist.topk(n_neighbors, largest=False).values
            score_1 = dist.view(int(dist.size(0)), -1)
            score_1 = torch.topk(score_1, 50, dim=1, largest=True)[0]
            score_1 = torch.mean(score_1, dim=1)
            loss = 0
        return loss, score_1

    def _init_centroid(self, model, data_loader):
        for i, (x, aug_img, _, _, _, _) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = model(x)
            phi_p, _ = self.Descriptor(p)
            self.scale = phi_p.size(2)
            self.C = ((self.C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i + 1)

class Descriptor(nn.Module):
    def __init__(self, classname):
        super(Descriptor, self).__init__()
        self.classname = classname
        if self.classname in ROTATION_OBJECTS:

            self.layer_3 = CoordConv2d(512, 512, 1)
            self.layer_1 = CoordConv2d(128, 128, 1)
            self.layer_2 = CoordConv2d(256, 256, 1)
        else:
            self.layer_1 = CoordConv2d(256, 256, 1)
            self.layer_2 = CoordConv2d(512, 512, 1)
            self.layer_3 = CoordConv2d(1024, 1024, 1)
        ##################################################
        self.conv1_1 = conv1x1(in_planes=512, out_planes=256)
        self.bn1_1 = nn.BatchNorm2d(256)
        self.relu1_1 = nn.ReLU()

        self.conv2_1 = conv1x1(in_planes=256, out_planes=256)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.relu2_1 = nn.ReLU()

        self.conv3_1 = conv1x1(in_planes=256, out_planes=256)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()

        ##################################################
        self.conv1_2 = conv1x1(in_planes=1024, out_planes=512)
        self.bn1_2 = nn.BatchNorm2d(512)
        self.relu1_2 = nn.ReLU()

        self.conv2_2 = conv1x1(in_planes=512, out_planes=512)
        self.bn2_2 = nn.BatchNorm2d(512)
        self.relu2_2 = nn.ReLU()

        self.conv3_2 = conv1x1(in_planes=512, out_planes=512)
        self.bn3_2 = nn.BatchNorm2d(512)
        self.relu3_2 = nn.ReLU()

        ##################################################

        self.conv1 = conv1x1(in_planes=256, out_planes=128)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = conv1x1(in_planes=128, out_planes=128)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = conv1x1(in_planes=128, out_planes=128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        ##################################################

    def forward(self, p):

        if self.classname in ROTATION_OBJECTS:
            p[0] = self.conv1(p[0])
            p[0] = self.bn1(p[0])
            p[0] = self.relu1(p[0])
            p[0] = self.conv2(p[0])
            p[0] = self.bn2(p[0])
            p[0] = self.relu2(p[0])
            p[0] = self.conv3(p[0])


            p[1] = self.conv1_1(p[1])
            p[1] = self.bn1_1(p[1])
            p[1] = self.relu1_1(p[1])
            p[1] = self.conv2_1(p[1])
            p[1] = self.bn2_1(p[1])
            p[1] = self.relu2_1(p[1])
            p[1] = self.conv3_1(p[1])


            p[2] = self.conv1_2(p[2])
            p[2] = self.bn1_2(p[2])
            p[2] = self.relu1_2(p[2])
            p[2] = self.conv2_2(p[2])
            p[2] = self.bn2_2(p[2])
            p[2] = self.relu2_2(p[2])
            p[2] = self.conv3_2(p[2])

        o_1 = F.avg_pool2d(p[0], 3, 1, 1)
        o_1 = self.layer_1(o_1)
        o_11 = F.interpolate(o_1, 64, mode='bilinear')
        o_2 =  F.avg_pool2d(p[1], 3, 1, 1)
        o_2 = self.layer_2(o_2)
        o_22 = F.interpolate(o_2, 64, mode='bilinear')
        o_3 = F.avg_pool2d(p[2], 3, 1, 1)
        o_3 = self.layer_3(o_3)
        o_33 = F.interpolate(o_3, 64, mode='bilinear')
        a = torch.cat((o_11, o_22), dim=1)
        phi_p = torch.cat((a, o_33), dim=1)
        return phi_p, p
