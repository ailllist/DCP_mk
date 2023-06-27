import torch
import numpy as np
from data import ModelNet40
from model import DCP
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

EPOCHS = 100

def train_one_epoch(net, train_loader, opt):
    net.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        rotations_ab.


def train(net, train_loader, test_loader):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    schedular = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1) # TODO 다시 보기

    # --------values--------------------
    best_test_loss = np.inf

if __name__ == "__main__":

    torch.cuda.empty_cache()
    train_loader = DataLoader(ModelNet40("train"),
                              shuffle=True, batch_size=2)
    test_loader = DataLoader(ModelNet40("test"),
                             shuffle=False, batch_size=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = DCP(512).to(device)

    for src, tgt, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in train_loader:

        src = src.to(device)
        tgt = tgt.to(device)

        model(src, tgt)
        break
