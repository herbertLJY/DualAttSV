import torch
from torch import nn
import torch.nn.init as init

class SelfPoolingDir(nn.Module):
    def __init__(self, input_num, output_num, feat_fc=None, inplanes=128):  # 2048,128
        super(SelfPoolingDir, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.inplanes = inplanes

        ## Linear_Q
        self.featQ = feat_fc

        ## Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, probe_value, probe_base):  # (bz/2)*sq*128; (bz/2)*sq*2048
        pro_size = probe_value.size()
        pro_batch = pro_size[0]  # 32
        pro_len = pro_size[1]  # 10

        ## generating Querys
        Qs = probe_base.view(pro_batch * pro_len, -1)  # 320*2048
        Qs = self.featQ(Qs)
        Qs = Qs.view(pro_batch, pro_len, -1)  # 32*10*128
        tmp_K = Qs
        Qmean = torch.mean(Qs, 1, keepdim=True)  # 32*1*128
        Hs = Qmean.expand(pro_batch, pro_len, self.output_num)  # 32*10*128

        weights = Hs * tmp_K  # 32*10*128
        weights = weights.permute(0, 2, 1)  # 32*128*10
        weights = weights.contiguous()

        weights = self.softmax(weights)

        weights = weights.permute(0, 2, 1)  # 32*10*128
        weights = weights.contiguous()
        pool_probe = probe_value * weights
        pool_probe = pool_probe.sum(1)

        return pool_probe
