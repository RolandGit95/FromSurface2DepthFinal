# %%

import os 
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from models import STLSTM
from datasets import BarkleyDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4
lr = 8e-4
val_split = 0.1
epochs = 16
hidden_size = 64

os.environ["WANDB_MODE"] = "dryrun"
wandb.init(project='FromSurface2DepthFinal', name='STLSTM_t32_d_0', reinit=True,dir="logs/")

# %%

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Training of Neural Networks, the Barkley Diver')
    parser.add_argument('-depth', '--depth', type=int)

    _args = parser.parse_args()
    args = vars(_args)
    

    #X = torch.rand(64,32,1,120,120).numpy()
    #Y = torch.rand(64,1,32,120,120).numpy()

    X = torch.load('data/X_train_debug.pt')#, map_location=device)
    Y = torch.load('data/Y_train_debug.pt')#, map_location=device)

    length = len(X)
    val_len = int(length*val_split)
    train_len = length - val_len

    model = nn.DataParallel(STLSTM(hidden_size=hidden_size)).to(device)#, device_ids=[0,1])
    #model.to(f'cuda:{model.device_ids[0]}') # .to(device)

    depth = args['depth']
    depths = [depth]
    train_dataset, val_dataset = torch.utils.data.random_split(BarkleyDataset(X, Y, depths=depths), [train_len, val_len])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=2, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=2, drop_last=True)

    val_dataloader_iter = iter(val_dataloader)
    criterion = nn.MSELoss()
    val_loss_fnc = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    callbacks = [ReduceLROnPlateau(optimizer, patience=512, factor=0.3, min_lr=1e-7, verbose=True)]

    torch.backends.cudnn.benchmark = True

    wandb.watch(model, log="all", log_freq=1)

    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']


    
    for epoch in range(epochs):
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataset)//batch_size):
            X = batch['X'].to(device)#.to(0)
            Y = batch['Y'].to(device)#.to(0)

            ##print(X.shape)
            for param in model.parameters():
                param.grad = None

            # forward + backward + optimize
            y_pred = model(X, max_depth=len(depths))
            loss = criterion(y_pred, Y[:,:,:])
            wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()

            del y_pred, loss

            wandb.log({"lr": get_lr()})


            if i % 16 == 0:
                try:
                    batch = next(val_dataloader_iter)
                    X_val, y_val = batch['X'], batch['Y']
                except StopIteration:
                    val_loader_iter = iter(val_dataloader)
                    batch = next(val_loader_iter)
                    X_val, y_val = batch['X'], batch['Y']
                X_val = X_val.to(0)
                y_val = y_val.to(0)

                with torch.no_grad():
                    val_outputs = model(X_val, max_depth=1)
                    val_loss = val_loss_fnc(y_val, val_outputs)
                    wandb.log({"val_loss": val_loss})

                for callback in callbacks:
                    callback.step(val_loss)

        torch.save(model, f'models/STLSTM_t32_d_{depth}_ep{epoch}')

                #for callback in callbacks:
                #    callback.step(val_loss)
        #print(X.shape)
    #with torch.no_grad():
    #    y_pred = model(X[:1], max_depth=32)


# %%


# %%
