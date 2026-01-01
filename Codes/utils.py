# -*- coding: utf-8 -*-
"""
By using the Code or the HydroLSTM representation in your publication(s), you agree to cite:

De la Fuente, L. A., Ehsani, M. R., Gupta, H. V., and Condon, L. E.: 
Towards Interpretable LSTM-based Modelling of Hydrological Systems,
EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-666, 2023.

"""

#from importing_MAC import * 
#from importing_notebook import * 
from importing import *
from datetime import timedelta
import torch
from tqdm import tqdm
import sys


#%%
def load_data(code, country, warm_up):
    PP, PET, Q = importing(code, country)
    Q['Q_obs'] = pd.to_numeric(Q['Q_obs'], errors='coerce')
    Q = Q[:PP.index[-1]]
    
    # Align start date based on warm_up and available Q data
    start_date = max(PP.index[0] + pd.DateOffset(days=warm_up), Q.dropna().index[0])
    return PP[start_date - pd.DateOffset(days=warm_up):], PET[start_date - pd.DateOffset(days=warm_up):], Q[start_date:]

#%%
class torch_dataset():
    def __init__(self, PP, PET, Q, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain):
        # Vectorized shifting for PP and PET
        pp_cols = [PP['PP'].shift(i).rename(f'PP_{i}' if i > 0 else 'PP') for i in range(lag)]
        pet_cols = [PET['PET'].shift(i).rename(f'PET_{i}' if i > 0 else 'PET') for i in range(lag)]
        X = pd.concat(pp_cols + pet_cols, axis=1).loc[ini_training - timedelta(days=1):].dropna()
        
        y = Q.loc[X.index, 'Q_obs'].values
        x = X.values
        
        if istrain:
            self.x_max, self.x_min, self.x_mean = x.max(0), x.min(0), x.mean(0)
            self.y_max, self.y_min, self.y_mean = y.max(), y.min(), y.mean()
        else:
            self.x_max, self.x_min, self.x_mean = x_max, x_min, x_mean
            self.y_max, self.y_min, self.y_mean = y_max, y_min, y_mean
                
        # Normalize and convert to torch tensors
        self.x = torch.from_numpy(((x - self.x_mean) / (self.x_max - self.x_min)).astype(np.float32))
        self.y = torch.from_numpy(((y - self.y_mean) / (self.y_max - self.y_min)).astype(np.float32))
        self.num_samples = len(self.y)

    def __len__(self): return self.num_samples   
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
    
#%%
def train_epoch(model, optimizer, loss_func, loader, epoch, loader_valid, patience, model_list, mean_valid_losses, DEVICE):
    model.train()
    train_losses = []
    with tqdm(loader, file=sys.stdout, leave=False, desc=f'# Epoch {epoch}') as pbar:
        for x, y in pbar:
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1, 1)
            model.epoch, model.DEVICE = epoch, DEVICE
            loss = loss_func(model(x)[0], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_losses.append(loss.item())
       
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for x, y in loader_valid:
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1, 1)
            valid_losses.append(loss_func(model(x)[0], y).item())

    mean_valid = np.mean(valid_losses)
    mean_valid_losses.append(mean_valid)
    
    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {np.mean(train_losses):.6f} | Valid Loss: {mean_valid:.6f}")
    
    model_list.append(model)     
    stopping = (epoch == patience)
    if stopping:
        print(f"Best model selected from epoch: {np.argmin(mean_valid_losses) + 1}")
        
    return stopping, model_list, mean_valid_losses
