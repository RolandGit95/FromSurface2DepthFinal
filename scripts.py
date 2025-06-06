# %%
import torch

X = torch.rand(64, 32, 1, 120, 120)
Y = torch.rand(64, 1, 32, 120, 120)

torch.save(X, "_X.npy")
torch.save(Y, "_Y.npy")
# %%
