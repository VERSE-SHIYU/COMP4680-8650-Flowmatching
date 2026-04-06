"""
Part 1.2: v-prediction flow matching at D=2
Train on all 3 datasets, generate samples via Euler ODE, plot comparison.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataloader import ToyDiffusionDataset, get_dataloader
from model import FlowMatchingMLP

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = ["swiss_roll", "gaussians", "circles"]
DIM = 2
BATCH_SIZE = 1024
LR = 1e-3
TRAIN_STEPS = 25000
EULER_STEPS = 50
NUM_SAMPLES = 2048  # how many samples to generate for visualization

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part1_generated"
CKPT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Training
# ============================================================
def train(dataset_name: str):
    """Train v-prediction flow matching on a single dataset at D=2."""
    print(f"\n{'='*50}")
    print(f"Training: {dataset_name}, D={DIM}")
    print(f"{'='*50}")

    dataloader = get_dataloader(name=dataset_name, dim=DIM, batch_size=BATCH_SIZE)
    model = FlowMatchingMLP(data_dim=DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    step = 0
    while step < TRAIN_STEPS:
        for x in dataloader:
            if step >= TRAIN_STEPS:
                break

            x = x.float().to(DEVICE)                          # (batch, D)
            batch_size = x.shape[0]

            # Sample t ~ Uniform(0, 1) and epsilon ~ N(0, I)
            t = torch.rand(batch_size, device=DEVICE)          # (batch,)
            eps = torch.randn_like(x)                          # (batch, D)

            # Construct noisy sample: zt = (1 - t) * x + t * eps
            t_expand = t.unsqueeze(-1)                         # (batch, 1)
            zt = (1 - t_expand) * x + t_expand * eps           # (batch, D)

            # Target velocity: v = eps - x
            v_target = eps - x                                 # (batch, D)

            # Forward pass
            v_pred = model(zt, t)                              # (batch, D)

            # Loss: MSE(v_pred, v_target)
            loss = nn.functional.mse_loss(v_pred, v_target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5000 == 0:
                print(f"  Step {step:>6d}/{TRAIN_STEPS} | Loss: {loss.item():.6f}")

            step += 1

    print(f"  Training complete. Final loss: {loss.item():.6f}")

    # Save checkpoint
    ckpt_path = CKPT_DIR / f"v_pred_v_loss_{dataset_name}_D{DIM}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    return model


# ============================================================
# Euler ODE Sampling
# ============================================================
@torch.no_grad()
def euler_sample(model, num_samples: int, dim: int, num_steps: int = EULER_STEPS):
    """
    Generate samples via Euler ODE integration.
    Start from z ~ N(0, I) at t=1, step toward t=0.
    """
    model.eval()

    # Initialize at t = 1 with pure noise
    z = torch.randn(num_samples, dim, device=DEVICE)
    dt = -1.0 / num_steps  # negative step: t goes from 1 to 0

    t_current = 1.0
    for _ in range(num_steps):
        t_batch = torch.full((num_samples,), t_current, device=DEVICE)
        v_pred = model(z, t_batch)
        z = z + v_pred * dt
        t_current += dt

    return z


# ============================================================
# Visualization
# ============================================================
def plot_comparison(real_2d, generated_2d, dataset_name, save_path):
    """Plot ground truth vs generated samples side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(real_2d[:, 0], real_2d[:, 1], s=1, alpha=0.5)
    axes[0].set_title(f"{dataset_name} — Ground Truth")
    axes[0].set_aspect("equal")

    axes[1].scatter(generated_2d[:, 0], generated_2d[:, 1], s=1, alpha=0.5, color="orange")
    axes[1].set_title(f"{dataset_name} — Generated (v-pred, D={DIM})")
    axes[1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Hyperparameters:")
    print(f"  Model: 6-layer MLP, 256 hidden units, 128-dim sinusoidal embedding")
    print(f"  Optimizer: Adam, lr={LR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Training steps: {TRAIN_STEPS}")
    print(f"  Euler sampling steps: {EULER_STEPS}")

    for name in DATASETS:
        # Train
        model = train(name)

        # Generate samples
        samples = euler_sample(model, NUM_SAMPLES, DIM)
        samples_np = samples.cpu().numpy()

        # Load ground truth for comparison
        ds = ToyDiffusionDataset(name=name, dim=DIM)
        real_np = ds.data.numpy()

        # Plot
        plot_comparison(
            real_2d=real_np,
            generated_2d=samples_np,
            dataset_name=name,
            save_path=OUTPUT_DIR / f"{name}_D{DIM}_v_pred.png",
        )


if __name__ == "__main__":
    main()
