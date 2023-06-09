import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from data import ModelNet40
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import numpy as np

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) # 단순히 module을 N번 반복한다.

def Attention(Q, K, V):

    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)

    return torch.matmul(p_attn, V), p_attn

def knn(x, k):
    # x: 32, 3, 1024
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # 32, 1024, 1024
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # norm의 제곱 # 32, 1, 1024
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # 가까운 점 반환
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.view(batch_size, -1, num_points)  # 32, 3 ,1024

    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # batch unpacking
    idx = idx.to(device)
    idx = idx + idx_base

    idx = idx.view(-1) # flatting

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    # 32768, 3을 가지고 [655360 (len of idx), 3]의 tensor를 만든다.

    feature = feature.view(batch_size, num_points, k, -1)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # 32, 1024, 20, 3을 만든다.

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # extract edge feature

    return feature

class DGCNN(nn.Module):

    def __init__(self, k=20, output=40, emb_dim=1024):
        super(DGCNN, self).__init__()
        self.k = k
        self.output = output
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(512, emb_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)  # 2, 64, 1024, 20
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=True)[0]


        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # 각 layer별 Feature를 모두 합친다.
        x = self.conv5(x)

        return torch.squeeze(x)


class PointNet(nn.Module):

    def __init__(self, emb_dims=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return  self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):  # Residual Connection
        return x + sublayer(self.norm(x))[0], sublayer(self.norm(x))[1]

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(F.relu(self.norm(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):

        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 2인 이유 : MultiATT -> FFN 이 2개의 과정으로 구성
        self.size = size

    def forward(self, x, mask):
        print("----x : ", x.shape)
        x, attn = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        pre_show = attn.cpu()
        pre_show = pre_show.detach().numpy()
        plt.imshow(pre_show[0][0])
        plt.show()
        return self.sublayer[1](x, self.feed_forward)[0]

class Encoder(nn.Module):

    def __init__(self, layer: "EncoderLayer", N):
        """

        :param layer: src_embed (EncoderLayer)
        :param N: self.N (args.n_blocks) -> Encoder를 몇번 반복할지?
        Encoder -> Encoder, 즉 MultiATT -> FFN -> MultiATT -> FFN구조로 만든다.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x, attn = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x, attn = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # print(x.shape)
        # pre_show = attn.cpu()
        # pre_show = pre_show.detach().numpy()
        # plt.imshow(pre_show[0][0])
        # plt.show()
        return self.sublayer[2](x, self.feed_forward)[0]


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)



class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):

        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        x = self.encoder(self.src_embed(src), src_mask)
        return x

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h:  # of head (쪼개는 갯수. default : 4)
        :param d_model: emb_dims
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 앞의 3개는 Linear Projection, 뒤 1개는 Universal Projection
        self.attn = None
        self.dropout = None

    def forward(self, quary, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = quary.size(0)

        # (2, 512, 1024) -> (2, 4, 1024, 128)
        quary, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (quary, key, value))]

        x, self.attn = Attention(quary, key, value)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)  # 2, 1024, 512
        # (batch_size, # of point, feature dims)
        print("----------------x.shape : ", x.shape)
        return self.linears[-1](x), self.attn



class Transformer(nn.Module):

    def __init__(self, emb_dims, n_blocks, n_heads, dropout, ff_dims):
        super(Transformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        c = copy.deepcopy
        attn = MultiHeadAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)  # emb_dims = 512, ff_dims = 1024
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class DCP(nn.Module):

    def __init__(self, emb_dims):
        super().__init__()
        self.emb_net = DGCNN(emb_dim=emb_dims)
        self.pointer = Transformer(emb_dims, 1, 4, 0.0, 1024)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_emb = self.emb_net(src)
        tgt_emb = self.emb_net(tgt)
        print("src : ", src.shape)
        print("embedded src : ", src_emb.shape)
        print("tgt : ", tgt.shape)
        print("embedded tgt : ", tgt_emb.shape)

        # ATT = Attention(src_emb, src_emb, src_emb)
        # pre_show = ATT[1].cpu()
        # pre_show = pre_show.detach().numpy()
        # plt.imshow(pre_show[0])
        # plt.show()
        src_emb_p, tgt_emb_p = self.pointer(src_emb, src_emb, src_emb)

        print("src_point : ", src_emb_p.shape)
        print("tgt_point : ", tgt_emb_p.shape)

        src_emb += src_emb_p
        tgt_emb += tgt_emb_p


if __name__ == "__main__":

    torch.cuda.empty_cache()
    train_loader = DataLoader(ModelNet40("train"), batch_size=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = DCP(512).to(device)

    for src, tgt, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in train_loader:

        src = src.to(device)
        tgt = tgt.to(device)

        model(src, tgt)
        break
