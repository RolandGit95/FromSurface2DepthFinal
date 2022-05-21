# %%

import os 
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from models import STLSTM
from datasets import BarkleyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 2
lr = 3e-4
val_split = 0.1
epochs = 2

os.environ["WANDB_MODE"] = "dryrun"
wandb.init(project='FromSurface2DepthFinal', name='STLSTM_t32_d32', reinit=True,dir="logs/")

# %%

def train(X, Y, model):
    dataset = BarkleyDataset(X, Y)

    model = nn.DataParallel(STLSTM(), device_ids=[0,1])
    model.to(f'cuda:{model.device_ids[0]}') # .to(device)

# %%

if __name__=='__main__':

    #X = torch.rand(64,32,1,120,120).numpy()
    #Y = torch.rand(64,1,32,120,120).numpy()

    X = torch.load('data/X_train_debug.pt')#, map_location=device)
    Y = torch.load('data/Y_train_debug.pt')#, map_location=device)

    length = len(X)
    val_len = int(length*val_split)
    train_len = length - val_len

    model = nn.DataParallel(STLSTM())

    train_dataset, val_dataset = torch.utils.data.random_split(BarkleyDataset(X, Y), [train_len, val_len])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=2)

    criterion = nn.MSELoss()
    val_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    torch.backends.cudnn.benchmark = True

    wandb.watch(model, log="all", log_freq=1)


    for epoch in range(epochs):
        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataset)):
            X = batch['X'].to(0)
            Y = batch['Y'].to(0)

            for param in model.parameters():
                param.grad = None

            # forward + backward + optimize
            y_pred = model(X, max_depth=1)
            loss = criterion(y_pred, Y[:,:,:1])
            wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()
        #print(X.shape)
    #with torch.no_grad():
    #    y_pred = model(X[:1], max_depth=32)


# %%


# %%
