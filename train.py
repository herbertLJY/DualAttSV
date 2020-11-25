import torch
import numpy as np
from resnet import resnet34
from attmodel import AttModule
from classifier import Classifier

num_classes = 1000
pre_bn = True
num_features = 256
cnn_model = resnet34(num_features=num_features, num_classes=num_classes, pre_bn=pre_bn)
cnn_model = cnn_model.cuda()

input_num = cnn_model.feat.in_features
att_model = AttModule(input_num, num_features, inplanes=128)
att_model = att_model.cuda()

binary_dropout = 0.5
classifier_model = Classifier(num_features, 1, drop=binary_dropout)
classifier_model = classifier_model.cuda()


def cos_dis(out_feat_p, out_feat_g):
    out_feat_p = torch.mean(out_feat_p, dim=1, keepdim=False)
    out_feat_g = torch.mean(out_feat_g, dim=1, keepdim=False)

    num_p = out_feat_p.size(0)
    num_g = out_feat_g.size(0)

    out_feat_p_norm = torch.norm(out_feat_p, p=2, dim=1)
    out_feat_g_norm = torch.norm(out_feat_g, p=2, dim=1)
    encode_scores = torch.matmul(out_feat_p, out_feat_g.permute(1, 0))
    encode_scores_norm = torch.matmul(out_feat_p_norm.view(num_p, 1), out_feat_g_norm.view(1, num_g))
    encode_scores = encode_scores / encode_scores_norm
    encodemat = encode_scores.view(num_p * num_g)
    single_distmat_all = encodemat.data.cpu().numpy()
    distmean = np.mean(single_distmat_all)
    return distmean


def train():
    cnn_model.train()
    att_model.train()
    classifier_model.train()

    batch_size, seq_len, mel_num = 64, 300, 64
    mel_input = torch.rand([batch_size, 1, seq_len, mel_num]).cuda()

    # We use pairwise sampler during training, so there should be 32 speakers in this example (batch 64),
    # each speaker provides 2 utterances.

    # Extract embeddings for all utterances
    feat, feat_cls, feat_raw = cnn_model(mel_input)
    # feat_cls is for classification loss

    # In attention model, they are separated into two parts, probe and gallery
    pooled_probe, pooled_gallery_2, pooled_probe_2, pooled_gallery = att_model(feat, feat_raw)
    # self-att utt1, mutual-att utt2, self-att utt2, mutual-att utt1

    encode_scores = classifier_model(pooled_probe, pooled_gallery_2, pooled_probe_2, pooled_gallery)
    # Because there are 32 speakers in one batch, this score matrix has size [32x32]
    return encode_scores


def eval():
    cnn_model.eval()
    att_model.eval()
    classifier_model.eval()

    seq_len, mel_num = 300, 64
    probe_seg, gallery_seg = 6, 8  # the segment numbers of test utterance and enrollment utterance
    mel_input_probe = torch.rand([probe_seg, 1, seq_len, mel_num]).cuda()
    mel_input_gallery = torch.rand([gallery_seg, 1, seq_len, mel_num]).cuda()

    with torch.no_grad():
        # Extract the embedding for test and enrollment utterances separately
        out_feat_p, _, out_raw_p = cnn_model(mel_input_probe)
        out_feat_g, _, out_raw_g = cnn_model(mel_input_gallery)

        cos_score = cos_dis(out_feat_p, out_feat_g)  # cosine distance

        # Apply attention layer
        pooled_probe = att_model.selfpooling_model(out_feat_p, out_raw_p)
        pooled_gallery = att_model.selfpooling_model(out_feat_g, out_raw_g)
        pooled_probe_mutual = att_model.mutualpooling_model(out_feat_p, out_raw_p, pooled_gallery)
        pooled_gallery_mutual = att_model.mutualpooling_model(out_feat_g, out_raw_g, pooled_probe)

        pooled_probe_mutual = pooled_probe_mutual.permute(1, 0, 2)
        pooled_probe, pooled_gallery = pooled_probe.unsqueeze(1), pooled_gallery.unsqueeze(0)
        encode_scores = classifier_model(pooled_probe, pooled_gallery_mutual, pooled_probe_mutual, pooled_gallery)
        encode_scores = torch.mean(encode_scores)
        encode_scores = encode_scores.data.cpu().numpy()

    return encode_scores, cos_score

t_s = train()
print(t_s.size())

test_b, test_c = eval()
print(test_b, test_c)



