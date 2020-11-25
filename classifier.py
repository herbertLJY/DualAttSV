import torch
from torch import nn
import torch.nn.init as init

class Classifier(nn.Module):
    def __init__(self, feat_num, class_num=1, drop=0):
        super(Classifier, self).__init__()
        self.feat_num = feat_num
        self.class_num = class_num
        self.drop = drop

        # BN layer
        self.classifierBN = nn.BatchNorm1d(self.feat_num)
        # feat classifeir
        self.classifierlinear = nn.Linear(self.feat_num, class_num)
        # dropout_layer
        self.drop = drop
        if self.drop > 0:
            self.droplayer = nn.Dropout(drop)

        init.constant_(self.classifierBN.weight, 1)
        init.constant_(self.classifierBN.bias, 0)

        init.normal_(self.classifierlinear.weight, std=0.001)
        init.constant_(self.classifierlinear.bias, 0)

    def forward(self, probe, gallery2, probe2, gallery):
        S_gallery2 = gallery2.size()
        N_probe = S_gallery2[0]
        N_gallery = S_gallery2[1]
        feat_num = S_gallery2[2]

        probe = probe.expand(N_probe, N_gallery, feat_num)
        gallery = gallery.expand(N_probe, N_gallery, feat_num)

        #    diff1, diff2 = probe - gallery2, probe2 - gallery
        diff1, diff2 = probe - gallery, probe2 - gallery2
        diff = diff1 * diff2
        pg_size = diff.size()
        p_size, g_size = pg_size[0], pg_size[1]
        diff = diff.view(p_size * g_size, -1)
        diff = diff.contiguous()
        diff = self.classifierBN(diff)
        if self.drop > 0:
            diff = self.droplayer(diff)
        cls_encode = self.classifierlinear(diff)
        cls_encode = cls_encode.view(p_size, g_size, -1)

        return torch.sigmoid(cls_encode)
