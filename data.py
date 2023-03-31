import os
import glob
import h5py
import numpy as np
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
    def __init__(self, partition="train", num_points=1024, num_of_object=-1, batch_size=1):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.data = self.data[:num_of_object, :self.num_points, :]
        self.label = self.label[:num_of_object, :]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]  # object의 갯수