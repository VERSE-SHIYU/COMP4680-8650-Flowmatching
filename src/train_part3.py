"""
Part 3: Can We Rescue v-Prediction?
====================================
swiss_roll D=32, 8 experiments (4 configs × 2 pred_types)

Experiment matrix:
    | Config              | v-pred + v-loss | x-pred + x-loss |
    |---------------------|-----------------|-----------------|
    | 256w,  25K steps    |       ✓         |       ✓         |
    | 256w, 100K steps    |       ✓         |       ✓         |
    | 512w,  50K steps    |       ✓         |       ✓         |
    | 1024w, 50K steps    |       ✓         |       ✓         |

Optimized for local GPU:
- Data pre-loaded to GPU once, no DataLoader
- Direct GPU tensor indexing per training step
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

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 8 experiments: (hidden_dim, total_steps, label)
CONFIGS = [
    (256,  25000,  "256w_25K"),
    (256,  100000, "256w_100K"),
    (512,  50000,  "512w_50K"),
    (1024, 50000,  "1024w_50K"),
]

PRED_TYPES = [
    ("v", "v"),  # v-pred + v-loss
    ("x", "x"),  # x-pred + x-loss
]


# ============================================================
# Model (supports configurable hidden_dim)
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t in [0,1] to fixed 128-dim embedding."""
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
    """
    6-layer MLP for flow matching with configurable width.
    Structure: [z_t; e_t] -> 5×(Linear+ReLU) -> Linear -> output (D-dim)
    """
    def __init__(self, data_dim, hidden_dim=256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=128)
        layers = []
        # Hidden layer 1: D+128 -> hidden_dim
        layers.extend([nn.Linear(data_dim + 128, hidden_dim), nn.ReLU()])
        # Hidden layers 2-5: hidden_dim -> hidden_dim
        for _ in range(4):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        # Output layer: hidden_dim -> D
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, zt, t):
        et = self.time_embed(t)
        x = torch.cat([zt, et], dim=-1)
        return self.net(x)


# ============================================================
# Training (GPU-optimized, no DataLoader)
# ============================================================
def train_model(all_data, pred_type, loss_type, hidden_dim, total_steps, tag=""):
    """
    Train flow matching model with data already on GPU.

    Args:
        all_data: (N, D) tensor on GPU
        pred_type: 'v' or 'x'
        loss_type: 'v' or 'x'
        hidden_dim: network width
        total_steps: number of training steps
        tag: label for logging
    """
    torch.manual_seed(SEED)
    model = FlowMatchingMLP(data_dim=DATA_DIM, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_total = all_data.shape[0]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params | hidden_dim={hidden_dim}")

    model.train()
    t_start = time.time()

    for step in range(total_steps):
        # Sample batch directly from GPU tensor
        idx = torch.randint(0, n_total, (BATCH_SIZE,), device=DEVICE)
        x = all_data[idx]

        # Sample t and epsilon
        t = torch.rand(BATCH_SIZE, device=DEVICE).clamp(EPS, 1 - EPS)
        eps = torch.randn_like(x)

        # Forward process: z_t = (1-t)*x + t*eps
        t_expand = t.unsqueeze(-1)
        zt = (1 - t_expand) * x + t_expand * eps

        # Model prediction
        output = model(zt, t)

        # Compute loss
        if pred_type == "v" and loss_type == "v":
            v_target = eps - x
            loss = nn.functional.mse_loss(output, v_target)
        elif pred_type == "v" and loss_type == "x":
            x_pred = zt - t_expand * output
            loss = nn.functional.mse_loss(x_pred, x)
        elif pred_type == "x" and loss_type == "x":
            loss = nn.functional.mse_loss(output, x)
        elif pred_type == "x" and loss_type == "v":
            v_pred = (zt - output) / t_expand
            v_target = eps - x
            loss = nn.functional.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10000 == 0 or step == total_steps - 1:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"    [{tag}] Step {step:>6d}/{total_steps} "
                f"| Loss: {loss.item():.6f} "
                f"| {steps_per_sec:.0f} steps/s"
            )

    elapsed = time.time() - t_start
    print(f"    Done in {elapsed:.1f}s. Final loss: {loss.item():.6f}")
    return model


# ============================================================
# Euler ODE Sampling
# ============================================================
@torch.no_grad()
def euler_sample(model, pred_type):
    """Euler ODE sampling from t=1 to t=0."""
    model.eval()
    z = torch.randn(NUM_SAMPLES, DATA_DIM, device=DEVICE)
    dt = -1.0 / EULER_STEPS

    t_current = 1.0
    for _ in range(EULER_STEPS):
        t_batch = torch.full((NUM_SAMPLES,), t_current, device=DEVICE)
        output = model(z, t_batch)

        if pred_type == "v":
            v = output
        else:
            t_clamped = max(t_current, EPS)
            v = (z - output) / t_clamped

        z = z + v * dt
        t_current += dt

    return z


# ============================================================
# Visualization
# ============================================================
def plot_comparison(real_2d, generated_2d, title_left, title_right, save_path):
    """Plot ground truth vs generated samples side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(real_2d[:, 0], real_2d[:, 1], s=1, alpha=0.5)
    axes[0].set_title(title_left)
    axes[0].set_aspect("equal")

    axes[1].scatter(
        generated_2d[:, 0], generated_2d[:, 1], s=1, alpha=0.5, color="orange"
    )
    axes[1].set_title(title_right)
    axes[1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_summary_grid(results, real_2d):
    """Plot 4×3 summary grid: configs (rows) × [GT, v-pred, x-pred] (cols)."""
    fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(18, 6 * len(CONFIGS)))

    for row, (hidden_dim, total_steps, config_label) in enumerate(CONFIGS):
        # Ground truth
        axes[row, 0].scatter(real_2d[:, 0], real_2d[:, 1], s=1, alpha=0.5)
        axes[row, 0].set_title("Ground Truth", fontsize=12)
        axes[row, 0].set_aspect("equal")
        axes[row, 0].set_ylabel(config_label, fontsize=14, fontweight="bold")

        # v-pred
        tag_v = f"v_pred_v_loss_{config_label}"
        if tag_v in results:
            s2d = results[tag_v]
            axes[row, 1].scatter(s2d[:, 0], s2d[:, 1], s=1, alpha=0.5, color="orange")
        axes[row, 1].set_title("v-pred + v-loss", fontsize=12)
        axes[row, 1].set_aspect("equal")

        # x-pred
        tag_x = f"x_pred_x_loss_{config_label}"
        if tag_x in results:
            s2d = results[tag_x]
            axes[row, 2].scatter(s2d[:, 0], s2d[:, 1], s=1, alpha=0.5, color="green")
        axes[row, 2].set_title("x-pred + x-loss", fontsize=12)
        axes[row, 2].set_aspect("equal")

    fig.suptitle(
        "Part 3: Rescuing v-Prediction on swiss_roll D=32",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    save_path = OUTPUT_DIR / "part3_summary_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary grid: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Running 8 experiments: 4 configs × 2 pred_types")
    print(f"Dataset: swiss_roll D=32")
    print(f"{'=' * 60}")

    # ---- Load data once, put on GPU ----
    dataset = ToyDiffusionDataset(name="swiss_roll", dim=32)
    all_data = dataset.data.float().to(DEVICE)
    real_2d = dataset.to_2d(dataset.data.numpy())
    print(f"Data loaded: shape={all_data.shape}, on {DEVICE}")
    print(f"Data range: [{all_data.min().item():.4f}, {all_data.max().item():.4f}]")

    # ---- Run experiments ----
    results = {}  # tag -> samples_2d (numpy)
    total_start = time.time()

    for hidden_dim, total_steps, config_label in CONFIGS:
        for pred_type, loss_type in PRED_TYPES:
            tag = f"{pred_type}_pred_{loss_type}_loss_{config_label}"
            print(f"\n{'=' * 60}")
            print(f"Experiment: {tag}")
            print(f"  hidden_dim={hidden_dim}, steps={total_steps}")
            print(f"{'=' * 60}")

            # Train
            model = train_model(
                all_data, pred_type, loss_type, hidden_dim, total_steps, tag=tag
            )

            # Sample
            samples = euler_sample(model, pred_type)
            samples_np = samples.cpu().numpy()
            samples_2d = dataset.to_2d(samples_np)
            results[tag] = samples_2d

            # Save individual plot
            save_path = OUTPUT_DIR / f"{tag}.png"
            plot_comparison(
                real_2d=real_2d,
                generated_2d=samples_2d,
                title_left="swiss_roll D=32 — Ground Truth",
                title_right=f"swiss_roll D=32 — {pred_type}-pred, {loss_type}-loss ({config_label})",
                save_path=save_path,
            )
            print(f"    Saved: {save_path}")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"All 8 experiments complete! Total time: {total_elapsed:.1f}s")

    # ---- Summary grid ----
    plot_summary_grid(results, real_2d)


if __name__ == "__main__":
    main()
