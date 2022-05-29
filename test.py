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
hidden_size = 128

os.environ["WANDB_MODE"] = "dryrun"

# %%

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Testing of Neural Networks, the Barkley Diver')
    parser.add_argument('-depth', '--depth', type=int)

    _args = parser.parse_args()
    args = vars(_args)
    

    #X = torch.rand(64,32,1,120,120).numpy()
    #Y = torch.rand(64,1,32,120,120).numpy()

    X = torch.load('data/X_test.pt')#, map_location=device)
    Y = torch.load('data/Y_test.pt')#, map_location=device)

    length = len(X)
    depth = args['depth']
    depths = [depth]

    model_path = os.path.join('models', f'STLSTM_t32_d_{depth}_ep14')

    model = nn.DataParallel(STLSTM(hidden_size=hidden_size)).to(device)#, device_ids=[0,1])
    model.load_state_dict(torch.load(model_path))

    #import IPython ; IPython.embed() ; exit(1)


    #wandb.init(project='FromSurface2DepthFinal', name=f'STLSTM_t32_d_{depth}', reinit=True,dir="logs/")

    test_dataset = BarkleyDataset(X, Y, depths=depths)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=2, drop_last=True)


    test_dataloader_iter = iter(test_dataloader)
    test_loss_fnc = nn.L1Loss()

    torch.backends.cudnn.benchmark = True

    LOSSES = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataset)//batch_size):
            X = batch['X'].to(device)#.to(0)
            Y = batch['Y'].to(device)#.to(0)

            y_pred = model(X, max_depth=len(depths))
            loss = test_loss_fnc(y_pred, Y[:,:,:]).detach().cpu().numpy()
            LOSSES.append(loss)
            print(loss)

    import IPython ; IPython.embed() ; exit(1)


        #torch.save(model, f'models/STLSTM_t32_d_{depth}_ep{epoch}')
        #torch.save(model.state_dict(), f'models/STLSTM_t32_d_{depth}_ep{epoch}')
                #for callback in callbacks:
                #    callback.step(val_loss)
        #print(X.shape)
    #with torch.no_grad():
    #    y_pred = model(X[:1], max_depth=32)


# %%


# %%
