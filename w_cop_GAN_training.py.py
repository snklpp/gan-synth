import os
import re
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
import ot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH    = "data.xlsx"
MODELS_DIR   = "models3"
OUTPUT_PATH  = "synthetic_data.xlsx"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 0
LATENT_DIM   = 10
HIDDEN_DIM   = 128
LR           = 5e-4
BETAS        = (0.5, 0.9)
N_EPOCHS     = 100_000
BATCH_SIZE   = 512
N_CRITIC     = 5
LAMBDA_GP    = 10.0
METRIC_FREQ  = 1  # compute & log metrics every METRIC_FREQ epochs

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─── Data Loading & CDF Utilities ─────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)

def fit_cdfs(df):
    """Return mappings (col → (xs, ps)) and their inverses for marginal CDF transforms."""
    forward, inverse = {}, {}
    for col in df.columns:
        vals = np.sort(df[col].values)
        uniq, idx  = np.unique(vals, return_index=True)
        ps = np.linspace(1/(len(vals)+1), len(vals)/(len(vals)+1), len(vals))
        # forward: x → p; inverse: p → x
        forward[col] = (uniq, ps[idx])
        inverse[col] = (ps[idx], uniq)
    return forward, inverse

def apply_cdf(df, forward_cdfs):
    """Map each column to a Gaussianized latent via its empirical CDF."""
    U = np.zeros_like(df.values, dtype=float)
    for i,col in enumerate(df.columns):
        xs, ps = forward_cdfs[col]
        U[:,i] = np.interp(df[col], xs, ps)
    U = np.clip(U, 1e-6, 1-1e-6)
    return norm.ppf(U)

def invert_samples(Z, inverse_cdfs):
    """Map Gaussian latent samples Z back to the original data space."""
    U = norm.cdf(Z)
    X = np.zeros_like(U)
    for i, col in enumerate(inverse_cdfs):
        ps, xs = inverse_cdfs[col]
        X[:,i] = np.interp(U[:,i], ps, xs)
    return X

# Precompute transforms
forward_cdfs, inverse_cdfs = fit_cdfs(df)
Z_real = apply_cdf(df, forward_cdfs).astype(np.float32)
dataset = TensorDataset(torch.from_numpy(Z_real))
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ─── Models ───────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )
    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 1),
        )
    def forward(self, x):
        return self.net(x)

# ─── Metric Functions ─────────────────────────────────────────────────────────
def calculate_emd(real, synth, num_bins=50):
    """1D EMD via histogram + ot.emd2."""
    real_hist, _ = np.histogram(real, bins=num_bins, density=True)
    synth_hist, _ = np.histogram(synth, bins=num_bins, density=True)
    real_hist  /= real_hist.sum()
    synth_hist /= synth_hist.sum()
    bin_centers = np.linspace(min(real.min(), synth.min()),
                              max(real.max(), synth.max()),
                              num_bins)
    cost = ot.dist(bin_centers.reshape(-1,1), bin_centers.reshape(-1,1))
    return float(ot.emd2(real_hist, synth_hist, cost))

def compute_metrics(G, noise, df, inverse_cdfs):
    """Return (MSE_corr, total_EMD) on full df."""
    with torch.no_grad():
        Zg = G(noise).cpu().numpy()
    syn = pd.DataFrame(invert_samples(Zg, inverse_cdfs), columns=df.columns)
    # Pearson corr MSE (exclude diagonal)
    rc = df.corr().values
    sc = syn.corr().values
    mask = ~np.eye(rc.shape[0], dtype=bool)
    mse_corr = np.mean((rc[mask] - sc[mask])**2)
    # sum of EMDs
    total_emd = sum(calculate_emd(df[col], syn[col]) for col in df.columns)
    return mse_corr, total_emd

# ─── Training Setup ──────────────────────────────────────────────────────────
G     = Generator().to(DEVICE)
D     = Critic().to(DEVICE)
optG  = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
optD  = optim.Adam(D.parameters(), lr=LR, betas=BETAS)
fixed_noise = torch.randn(len(df), LATENT_DIM, device=DEVICE)

def gradient_penalty(critic, real, fake):
    B = real.size(0)
    α = torch.rand(B, 1, device=DEVICE)
    x̂ = (α * real + (1-α) * fake).requires_grad_(True)
    d̂ = critic(x̂)
    grads = torch.autograd.grad(d̂.sum(), x̂, create_graph=True)[0]
    grad_norm = grads.view(B, -1).norm(2, dim=1)
    return ((grad_norm - 1)**2).mean()

# ─── Training Loop ───────────────────────────────────────────────────────────
best_score = float('inf')

for epoch in range(1, N_EPOCHS+1):
    for real_batch_tuple in loader:
        real = real_batch_tuple[0].to(DEVICE)
        # --- Update Critic ---
        for _ in range(N_CRITIC):
            z = torch.randn(real.size(0), LATENT_DIM, device=DEVICE)
            fake = G(z).detach()
            d_loss = D(fake).mean() - D(real).mean() + LAMBDA_GP * gradient_penalty(D, real, fake)
            optD.zero_grad(); d_loss.backward(); optD.step()

        # --- Update Generator ---
        z = torch.randn(real.size(0), LATENT_DIM, device=DEVICE)
        g_loss = -D(G(z)).mean()
        optG.zero_grad(); g_loss.backward(); optG.step()

    # --- Logging & Checkpointing ---
    if epoch % METRIC_FREQ == 0:
        print(f"[{epoch:5d}] D_loss={d_loss:.4f}  G_loss={g_loss:.4f}")
        path = os.path.join(MODELS_DIR, f"G_epoch{epoch}.pth")
        torch.save(G.state_dict(), path)
        print("  → Saved model at epoch", epoch)

# ─── Generate & Save Synthetic Data ──────────────────────────────────────────
G.load_state_dict(torch.load(path, map_location=DEVICE))
G.eval()
with torch.no_grad():
    Zg = G(fixed_noise).cpu().numpy()
X_syn = invert_samples(Zg, inverse_cdfs)
synthetic_df = pd.DataFrame(X_syn, columns=df.columns)
synthetic_df.to_csv(OUTPUT_PATH.replace('.xlsx', '.csv'), index=False)
print("Done — synthetic data saved to", OUTPUT_PATH.replace('.xlsx', '.csv'))
