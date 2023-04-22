import os
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    all_data = []
    all_label = []

    for h5_name in glob.glob(os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048", "ply_data_%s*.h5" % partition)):
        with h5py.File(h5_name) as f:
            data = f["data"][:]
            label = f["label"][:]
            all_data.append(data)
            all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

class ModelNet40(Dataset):
    def __init__(self, partition="train", num_points=1024, num_of_object=-1, batch_size=1, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.data = self.data[:num_of_object, :self.num_points, :]
        self.label = self.label[:num_of_object, :]
        self.factor = factor

    def __getitem__(self, index):
        pointcloud = self.data[index][:self.num_points]
        anglex = np.random.uniform() * np.pi / self.factor  # dataset 생성을 위한 x
        angley = np.random.uniform() * np.pi / self.factor  # dataset 생성을 위한 y
        anglez = np.random.uniform() * np.pi / self.factor  # dataset 생성을 위한 z

        cosx = np.cos(anglex)  # 변환 행렬 만들기
        cosy = np.cos(angley)  # 변환 행렬 만들기
        cosz = np.cos(anglez)  # 변환 행렬 만들기
        sinx = np.sin(anglex)  # 변환 행렬 만들기
        siny = np.sin(angley)  # 변환 행렬 만들기
        sinz = np.sin(anglez)  # 변환 행렬 만들기
        Rx = np.array([[1, 0, 0],
                      [0, cosx, -sinx],
                      [0, sinx, cosx]])  # x축 변환 행렬
        Ry = np.array([[cosy, 0, siny],
                      [0, 1, 0],
                      [-siny, 0, cosy]])  # y축 변환 행렬
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])  # z축 변환 행렬

        R_ab = Rx.dot(Ry).dot(Rz)  # 회전 변환 행렬
        R_ba = R_ab.T

        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler("zyx", [anglez, angley, anglex])  # extrinsic rotation
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype("float32"), pointcloud2.astype("float32"), R_ab.astype("float32"), \
                translation_ab.astype("float32"), R_ba.astype("float32"), translation_ba.astype("float32"), \
                euler_ba.astype("float32"), euler_ba.astype("float32")

    def __len__(self):
        return self.data.shape[0]  # object의 갯수

if __name__ == "__main__":
    train_data = ModelNet40("train")
    test_data = ModelNet40("test")

    print(train_data[0])
