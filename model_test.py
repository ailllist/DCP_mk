import torch
import numpy as np
from data import ModelNet40
from model import DCP
from torch.utils.data import DataLoader

if __name__ == "__main__":
    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True

    test_data = DataLoader(ModelNet40("test"), batch_size=2, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = DCP(512).to(device)
    model.eval()
    model.load_state_dict(torch.load("model.best.t7"), strict=False)

    for src, tgt, rot_ab, trans_ab, rot_ba, trans_ba, euler_ab, euler_ba in test_data:
        src = src.to(device)
        tgt = tgt.to(device)
        rot_ab_pred, trans_ab_pred, rot_ba_pred, trans_ba_pred = model(src, tgt)

        print("rot_ab : ", rot_ab)
        print("rot_ab_pred : ", rot_ab_pred)
        print("rot_ba : ", rot_ba)
        print("rot_ba_pred : ", rot_ba_pred)

        print("trans_ab : ", trans_ab)
        print("trans_ab_pred : ", trans_ab_pred)
        print("trans_ba : ", trans_ba)
        print("trans_ba_pred : ", trans_ba_pred)

        break