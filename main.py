import torch
import numpy as np
from data import ModelNet40
from model import DCP
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import tqdm


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


