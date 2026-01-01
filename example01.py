import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
import numpy as np
import pandas as pd

from Hydro_LSTM import Model_hydro_lstm
from LSTM import Model_lstm
from utils import load_data, torch_dataset, train_epoch

# --- Configuration ---
code = 11523200
model_option = "HYDRO"
# model_option = "LSTM"
cells = 1
memory = 64
epochs = 20
learning_rate = 1e-3
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for consistent results

print(f"Running {model_option} for catchment {code} on {device}")

# --- Data Loading & Preparation ---
lag = memory + 1
PP, PET, Q = load_data(code, 'US', lag)

ini_train = PP.index[0] + timedelta(days=275)
end_train, end_valid = 7303, 7303 + 1462

# Create datasets using training statistics for validation
ds = torch_dataset(PP.loc[:ini_train + timedelta(days=end_train)], 
                   PET.loc[:ini_train + timedelta(days=end_train)], 
                   Q.loc[:ini_train + timedelta(days=end_train)], 
                   lag, ini_train, [], [], [], [], [], [], istrain=True)

ds_valid = torch_dataset(PP.loc[:ini_train + timedelta(days=end_valid-1)], 
                         PET.loc[:ini_train + timedelta(days=end_valid-1)], 
                         Q.loc[:ini_train + timedelta(days=end_valid-1)], 
                         lag, ini_train, ds.x_max, ds.x_min, ds.x_mean, ds.y_max, ds.y_min, ds.y_mean, istrain=False)

# Slice validation set to post-training period
ds_valid.x, ds_valid.y = ds_valid.x[end_train+1:], ds_valid.y[end_train+1:]
ds_valid.num_samples = len(ds_valid.y)

# Dataloaders
is_hydro = (model_option == "HYDRO")
loader = DataLoader(ds, batch_size=batch_size, shuffle=not is_hydro, 
                    sampler=SequentialSampler(ds) if is_hydro else None)
loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=not is_hydro,
                          sampler=SequentialSampler(ds_valid) if is_hydro else None)

# --- Model Setup ---
input_size = 2 * lag # n_variables * lag
model_cls = Model_hydro_lstm if is_hydro else Model_lstm
model = model_cls(input_size if is_hydro else 2, lag, cells, 0.0).to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.SmoothL1Loss()

# --- Training ---
valid_losses, model_list = [], []
print("Starting training...")

# Save preprocessed data for Julia comparison
np.savez('preprocessed_data.npz', 
         X_train=ds.x.numpy(), 
         Y_train=ds.y.numpy(), 
         X_valid=ds_valid.x.numpy(), 
         Y_valid=ds_valid.y.numpy(),
         y_stats=np.array([ds.y_max, ds.y_min, ds.y_mean]))
print("Preprocessed data saved to 'preprocessed_data.npz' for Julia.")

for epoch in range(1, epochs + 1):
    stop, model_list, valid_losses = train_epoch(model, optimizer, loss_func, loader, epoch, 
                                                 loader_valid, epochs, model_list, valid_losses, device)
    if stop: break
        
model = model_list[np.argmin(valid_losses)]

# --- Evaluation ---
model.eval()
with torch.no_grad():
    model.epoch, model.DEVICE = 0, device
    pred, _, _ = model(ds_valid.x.to(device))
    
    # Inverse transform and metrics
    pred_np = pred.cpu().numpy().flatten() * (ds.y_max - ds.y_min) + ds.y_mean
    obs_np = ds_valid.y.numpy().flatten() * (ds.y_max - ds.y_min) + ds.y_mean
    
    rmse = np.sqrt(mean_squared_error(obs_np, pred_np))
    print(f"\nResults: RMSE={rmse:.4f}, MAE={mean_absolute_error(obs_np, pred_np):.4f}, R2={r2_score(obs_np, pred_np):.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(obs_np[:200], label='Observed', alpha=0.7)
    plt.plot(pred_np[:200], label='Predicted', alpha=0.7)
    plt.title(f"Catchment {code} - {model_option} Model")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('example01_results.png')
    print("Plot saved as 'example01_results.png'")
