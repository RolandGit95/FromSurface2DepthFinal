import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

# %%


class BarkleyDataset(Dataset):
    def __init__(self, X, Y, depths=np.arange(0,32,1), time_steps=np.arange(0,32,1)) -> None:

        super(BarkleyDataset, self).__init__()

        self.transform = lambda data: (data.float()+127)/255.
        self.target_transform = lambda data: (data.float()+127)/255.

        self.X = X[:,time_steps] #dimensions: [N,T,D,H,W]#torch.from_numpy(X)
        self.Y = Y[:,:,depths] #torch.from_numpy(Y)

    def __getitem__(self, idx: int):

        # transform data of type int8 to float32 only at execution time to save memory
        X, Y = self.transform(self.X[idx]), self.target_transform(self.Y[idx])

        # Training data augmentation (random rotation of 0,90,180 or 270 degree)
        k = np.random.randint(0, 4)
        X = torch.rot90(X, k=k, dims=[2, 3])
        Y = torch.rot90(Y, k=k, dims=[2, 3])

        return {'X': X, 'Y': Y}

    def __len__(self):
        return len(self.X)