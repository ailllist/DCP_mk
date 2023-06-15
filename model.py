import math

import torch
from torch import nn
import torch.nn.functional as F
from data import ModelNet40
import torch.optim as optim
from torch.utils.data import DataLoader
import math

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


class DCP(nn.Module):

    def __init__(self, emb_dims):
        super().__init__()
        self.emb_net = DGCNN(emb_dim=emb_dims)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_emb = self.emb_net(src)
        tgt_emb = self.emb_net(tgt)
        print("src : ", src.shape)
        print("embedded src : ", src_emb.shape)
        print("tgt : ", tgt.shape)
        print("embedded tgt : ", tgt_emb.shape)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]

        src_emb = self.emb_net(src)
        tgt_emb = self.emb_net(tgt)
        print(src_emb.shape)
        # self-attention
        src_att = Attention(src_emb, src_emb, src_emb)
        tgt_att = Attention(tgt_emb, tgt_emb, tgt_emb)

        print(src_att[0])
        print(src_att[0].shape)
        print(tgt_att[0].shape)

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
