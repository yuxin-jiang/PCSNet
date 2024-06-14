import torch
from torch import nn
import torch.nn.functional as F

import resnet as models


# Masked Average Pooling
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        layers = 50
        classes = 1
        pretrained = True
        assert layers in [50, 101, 152]
        # assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        # self.shot = args.shot
        self.train_iter = True
        # self.eval_iter = args.eval_iter
        self.pyramid = True

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        reduce_dim = 896
        # reduce_dim = 384
        fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        # Using Feature Enrichment Module from PFENet as context module
        if self.pyramid:

            self.pyramid_bins = [60, 30, 15, 8]
            self.avgpool_list = []

            for bin in self.pyramid_bins:
                if bin > 1:
                    self.avgpool_list.append(
                        nn.AdaptiveAvgPool2d(bin)
                    )

            self.corr_conv = []
            self.beta_conv = []
            self.inner_cls = []

            for bin in self.pyramid_bins:
                self.corr_conv.append(nn.Sequential(
                    nn.Conv2d(reduce_dim+200, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                ))
                self.beta_conv.append(nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                ))
                self.inner_cls.append(nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),
                    nn.Conv2d(reduce_dim, classes, kernel_size=1),
                ))
            self.corr_conv = nn.ModuleList(self.corr_conv)
            self.beta_conv = nn.ModuleList(self.beta_conv)
            self.inner_cls = nn.ModuleList(self.inner_cls)

            self.alpha_conv = []
            for idx in range(len(self.pyramid_bins) - 1):
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(2 * reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                ))
            self.alpha_conv = nn.ModuleList(self.alpha_conv)

            self.res1 = nn.Sequential(
                nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.res2 = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

    def forward(self, guide_feat,prob_map, y=None):

########################### Context Module ###########################
        if self.pyramid:

            out_list = []
            pyramid_feat_list = []

            for idx, tmp_bin in enumerate(self.pyramid_bins):
                if tmp_bin <= 1.0:
                    bin = int(guide_feat.shape[2] * tmp_bin)
                    guide_feat_bin = nn.AdaptiveAvgPool2d(bin)(guide_feat)
                else:
                    bin = tmp_bin
                    guide_feat_bin = self.avgpool_list[idx](guide_feat)
                prob_map_bin = F.interpolate(prob_map, size=(bin, bin), mode='bilinear', align_corners=True)
                merge_feat_bin = torch.cat([guide_feat_bin, prob_map_bin], 1)
                merge_feat_bin = self.corr_conv[idx](merge_feat_bin)

                if idx >= 1:
                    pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                    pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                    rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                    merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

                merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
                inner_out_bin = self.inner_cls[idx](merge_feat_bin)
                merge_feat_bin = F.interpolate(merge_feat_bin, size=(guide_feat.size(2), guide_feat.size(3)),
                                               mode='bilinear', align_corners=True)
                pyramid_feat_list.append(merge_feat_bin)
                out_list.append(inner_out_bin)

            final_feat = torch.cat(pyramid_feat_list, 1)
            final_feat = self.res1(final_feat)
            final_feat = self.res2(final_feat) + final_feat
            out = self.cls(final_feat)

            out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=True)
            return out