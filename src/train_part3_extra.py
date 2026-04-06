"""
Part 3 追加实验: v-pred 1024w 100K steps
只跑这一个实验，约10分钟
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from pathlib import Path
import time

from dataloader import ToyDiffusionDataset

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
LR = 1e-3
EULER_STEPS = 50
NUM_SAMPLES = 2048
DATA_DIM = 32
EPS = 1e-5
SEED = 42
HIDDEN_DIM = 1024
TOTAL_STEPS = 100000

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        k = embed_dim // 2
        freqs = torch.exp(
            -torch.arange(k, dtype=torch.float32) * math.log(10000) / (k - 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim=256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=128)
        layers = []
        layers.extend([nn.Linear(data_dim + 128, hidden_dim), nn.ReLU()])
        for _ in range(4):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, zt, t):
        et = self.time_embed(t)
        x = torch.cat([zt, et], dim=-1)
        return self.net(x)


# ============================================================
# Main
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Experiment: v-pred 1024w 100K steps")
    print(f"{'=' * 60}")

    # Load data
    dataset = ToyDiffusionDataset(name="swiss_roll", dim=32)
    all_data = dataset.data.float().to(DEVICE)
    real_2d = dataset.to_2d(dataset.data.numpy())
    print(f"Data loaded: {all_data.shape}")

    # Train
    torch.manual_seed(SEED)
    model = FlowMatchingMLP(data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_total = all_data.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    model.train()
    t_start = time.time()

    for step in range(TOTAL_STEPS):
        idx = torch.randint(0, n_total, (BATCH_SIZE,), device=DEVICE)
        x = all_data[idx]

        t = torch.rand(BATCH_SIZE, device=DEVICE).clamp(EPS, 1 - EPS)
        eps = torch.randn_like(x)
        t_expand = t.unsqueeze(-1)
        zt = (1 - t_expand) * x + t_expand * eps

        v_target = eps - x
        output = model(zt, t)
        loss = nn.functional.mse_loss(output, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10000 == 0 or step == TOTAL_STEPS - 1:
            elapsed = time.time() - t_start
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  Step {step:>6d}/{TOTAL_STEPS} | Loss: {loss.item():.6f} | {sps:.0f} steps/s")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s")

    # Sample
    model.eval()
    z = torch.randn(NUM_SAMPLES, DATA_DIM, device=DEVICE)
    dt = -1.0 / EULER_STEPS
    t_current = 1.0
    with torch.no_grad():
        for _ in range(EULER_STEPS):
            t_batch = torch.full((NUM_SAMPLES,), t_current, device=DEVICE)
            v = model(z, t_batch)
            z = z + v * dt
            t_current += dt

    samples_2d = dataset.to_2d(z.cpu().numpy())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(real_2d[:, 0], real_2d[:, 1], s=1, alpha=0.5)
    axes[0].set_title("swiss_roll D=32 — Ground Truth")
    axes[0].set_aspect("equal")
    axes[1].scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, color="orange")
    axes[1].set_title("swiss_roll D=32 — v-pred, v-loss (1024w_100K)")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    save_path = OUTPUT_DIR / "v_pred_v_loss_1024w_100K.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
