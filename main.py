import gc
import torch
import numpy as np
from data import ModelNet40
from model import DCP
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from util import transform_point_cloud, npmat2euler
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

TRAIN_EPOCHS = 100

def test_one_epoch(net, test_loader):
    net.eval()

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

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(
            test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        # save rot & tl of ab
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())

        # save rot & tl of ba
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        # 예측값으로 src point cloud를 돌려본다.
        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
        # 예측값으로 target point cloud를 돌려본다.

        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)
        # Rotation matrix는 Orthogonal하기 때문에 A^TA = I를 만족할 때까지 계속 학습...

        total_loss += loss.item() * batch_size

        mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size  # 모든 원소의 평균
        mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)  # 고차원에서의 append
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, translations_ab, \
        rotations_ab_pred, translations_ab_pred, rotations_ba, translations_ba, rotations_ba_pred, \
        translations_ba_pred, eulers_ab, eulers_ba

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
        opt.zero_grad() #TODO 설마 이거때문??????
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        # save rot & tl of ab
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())

        # save rot & tl of ba
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        # 예측값으로 src point cloud를 돌려본다.
        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
        # 예측값으로 target point cloud를 돌려본다.

        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                + F.mse_loss(translation_ab_pred, translation_ab)
        # Rotation matrix는 Orthogonal하기 때문에 A^TA = I를 만족할 때까지 계속 학습...'
        # loss.requires_grad_(True)
        # print("loss : ", loss)
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

        mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size  # 모든 원소의 평균
        mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)  # 고차원에서의 append
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples,\
        mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, translations_ab,\
        rotations_ab_pred, translations_ab_pred, rotations_ba, translations_ba, rotations_ba_pred, \
        translations_ba_pred, eulers_ab, eulers_ba



def train(net, train_loader, test_loader):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1) # TODO 다시 보기
    f = open("run.csv", "w")
    f.close()
    # --------------------init value--------------------------
    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_test_mse_ba = np.inf
    best_test_rmse_ba = np.inf
    best_test_mae_ba = np.inf

    best_test_r_mse_ba = np.inf
    best_test_r_rmse_ba = np.inf
    best_test_r_mae_ba = np.inf
    best_test_t_mse_ba = np.inf
    best_test_t_rmse_ba = np.inf
    best_test_t_mae_ba = np.inf\

    for epoch in range(TRAIN_EPOCHS):
        print(epoch)
        scheduler.step()
        train_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred, train_eulers_ab, train_eulers_ba = train_one_epoch(net, train_loader, opt)

        test_loss, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(net, test_loader)

        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        train_rmse_ba = np.sqrt(train_mse_ba)
        test_rmse_ba = np.sqrt(test_mse_ba)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred, 'xyz')
        train_r_mse_ba = np.mean((train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)) ** 2)
        train_r_rmse_ba = np.sqrt(train_r_mse_ba)
        train_r_mae_ba = np.mean(np.abs(train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)))
        train_t_mse_ba = np.mean((train_translations_ba - train_translations_ba_pred) ** 2)
        train_t_rmse_ba = np.sqrt(train_t_mse_ba)
        train_t_mae_ba = np.mean(np.abs(train_translations_ba - train_translations_ba_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

        test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
        test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
        test_r_rmse_ba = np.sqrt(test_r_mse_ba)
        test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
        test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
        test_t_rmse_ba = np.sqrt(test_t_mse_ba)
        test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))
        with open("run.csv", "a") as f:
            f.write(f"{test_loss},{test_r_mse_ab},{test_r_mae_ab},{test_t_mse_ab},{test_t_mae_ab},"
                    f"{test_r_mse_ba},{test_r_mae_ba},{test_t_mse_ba},{test_t_mae_ba},"
                    f"{test_mse_ab},{test_mae_ab},{test_mse_ba},{test_mae_ba}\n")

        if best_test_loss >= test_loss:
            best_test_loss = test_loss

            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_test_mse_ba = test_mse_ba
            best_test_rmse_ba = test_rmse_ba
            best_test_mae_ba = test_mae_ba

            best_test_r_mse_ba = test_r_mse_ba
            best_test_r_rmse_ba = test_r_rmse_ba
            best_test_r_mae_ba = test_r_mae_ba

            best_test_t_mse_ba = test_t_mse_ba
            best_test_t_rmse_ba = test_t_rmse_ba
            best_test_t_mae_ba = test_t_mae_ba

            torch.save(net.state_dict(), "model.best.t7")
        torch.save(net.state_dict(), "model.best.t7")
        gc.collect()

if __name__ == "__main__":

    SEED = 1234
    batch_size = 8
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(ModelNet40("train"),
                              shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(ModelNet40("test"),
                             shuffle=False, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = DCP(512).to(device)

    train(model, train_loader, test_loader)

