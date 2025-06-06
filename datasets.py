import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset used to load Barkley simulation data

__all__ = ["BarkleyDataset"]

# %%


class BarkleyDataset(Dataset):
    """Torch dataset handling Barkley simulation data.

    Parameters
    ----------
    X : torch.Tensor
        Input samples with shape ``[N, T, D, H, W]``.
    Y : torch.Tensor
        Ground truth with shape ``[N, 1, D, H, W]``.
    depths : array-like, optional
        Depth indices to select from ``Y``.
    time_steps : array-like, optional
        Time steps to select from ``X``.
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        depths: np.ndarray = np.arange(0, 32, 1),
        time_steps: np.ndarray = np.arange(0, 32, 1),
    ) -> None:

        super().__init__()

        self.transform = lambda data: (data.float() + 127) / 255.0
        self.target_transform = lambda data: (data.float() + 127) / 255.0

        # Select given time steps and depths
        self.X = X[:, time_steps]
        self.Y = Y[:, :, depths]

    def __getitem__(self, idx: int):
        """Return a sample from the dataset."""

        # convert on the fly to save memory
        X = self.transform(self.X[idx])
        Y = self.target_transform(self.Y[idx])

        # random rotation by 0, 90, 180 or 270 degrees
        k = np.random.randint(0, 4)
        X = torch.rot90(X, k=k, dims=[2, 3])
        Y = torch.rot90(Y, k=k, dims=[2, 3])

        return {"X": X, "Y": Y}

    def __len__(self):
        return len(self.X)
