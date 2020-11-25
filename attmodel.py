from torch import nn
from selfatt import SelfPoolingDir
from mutualatt import mutualPoolingDir
import torch.nn.init as init


class AttModule(nn.Module):
    def __init__(self, input_num, output_num, dropout=0, inplanes=128, **unused):  # 2048 ,128
        super(AttModule, self).__init__()

        self.input_num = input_num
        self.output_num = output_num
        self.inplanes = inplanes

        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

        ## attention modules
        self.feat_fc = nn.Sequential(nn.Linear(self.input_num, self.inplanes),
                                     nn.BatchNorm1d(self.inplanes),
                                     nn.Linear(self.inplanes, self.output_num),
                                     nn.BatchNorm1d(self.output_num))
        for m in self.feat_fc.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            else:
                print(type(m))

        self.selfpooling_model = SelfPoolingDir(self.input_num, self.output_num, feat_fc=self.feat_fc)
        self.mutualpooling_model = mutualPoolingDir(self.input_num, self.output_num, feat_fc=self.feat_fc)

    def forward(self, x, raw_x):  # x(bz*sq*128) input(bz*sq*2048)
        xsize = x.size()
        sample_num, seq_len = xsize[0], xsize[1]  # 64, 10

        if sample_num % 2 != 0:
            raise RuntimeError("the batch size should be even number!")

        x = x.view(int(sample_num / 2), 2, seq_len, -1)  # 32*2*10*128

        if (self.dropout_layer is not None) and self.training:
            raw_x = raw_x.view(sample_num * seq_len, -1)  # 32*2*10*2048
            raw_x = self.dropout_layer(raw_x)
            raw_x = raw_x.view(sample_num, seq_len, -1)

        x = x.view(int(sample_num / 2), 2, seq_len, -1)  # 32*2*10*128
        raw_x = raw_x.view(int(sample_num / 2), 2, seq_len, -1)  # 32*2*10*2048
        probe_x = x[:, 0, :, :]  # 32*10*128
        probe_x = probe_x.contiguous()
        gallery_x = x[:, 1, :, :]  # 32*10*128
        gallery_x = gallery_x.contiguous()

        probe_input = raw_x[:, 0, :, :]  # 32*10*2048
        probe_input = probe_input.contiguous()
        gallery_input = raw_x[:, 1, :, :]  # 32*10*2048
        gallery_input = gallery_input.contiguous()

        ## self-pooling
        pooled_probe = self.selfpooling_model(probe_x, probe_input)
        pooled_gallery = self.selfpooling_model(gallery_x, gallery_input)

        ## mutual-pooling
        # gallery_x(32*10*128), gallery_input(32*10*2048), pooled_probe(32*128)
        pooled_gallery_2 = self.mutualpooling_model(gallery_x, gallery_input, pooled_probe)
        pooled_probe_2 = self.mutualpooling_model(probe_x, probe_input, pooled_gallery)

        pooled_probe_2 = pooled_probe_2.permute(1, 0, 2)
        pooled_probe, pooled_gallery = pooled_probe.unsqueeze(1), pooled_gallery.unsqueeze(0)
        # 32*1*128, 32*32*128, 32*32*128, 1*32*128
        return pooled_probe, pooled_gallery_2, pooled_probe_2, pooled_gallery  # (bz/2) * 128,  (bz/2)*(bz/2)*128
