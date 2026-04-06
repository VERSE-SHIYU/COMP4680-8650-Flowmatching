"""
Part 2: Flow Matching Parameterization
4 combinations (v-pred/v-loss, v-pred/x-loss, x-pred/x-loss, x-pred/v-loss)
× 3 datasets × 3 dimensions = 36 experiments
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
DIMS = [2, 8, 32]
BATCH_SIZE = 1024
LR = 1e-3
TRAIN_STEPS = 25000
EULER_STEPS = 50
NUM_SAMPLES = 2048
EPS = 1e-5  # clip t to [EPS, 1-EPS] to avoid division by zero

# 4 combinations: (prediction_type, loss_type)
COMBINATIONS = [
    ("v", "v"),  # v-prediction + v-loss
    ("v", "x"),  # v-prediction + x-loss
    ("x", "x"),  # x-prediction + x-loss
    ("x", "v"),  # x-prediction + v-loss
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part2"
CKPT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints_part2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Training
# ============================================================
def train(dataset_name: str, dim: int, pred_type: str, loss_type: str):
    """
    Train flow matching with specified prediction and loss types.

    pred_type: "v" or "x" — what the model outputs
    loss_type: "v" or "x" — which space to compute loss in
    """
    tag = f"{pred_type}_pred_{loss_type}_loss_{dataset_name}_D{dim}"
    print(f"\n  Training: {tag}")

    dataloader = get_dataloader(name=dataset_name, dim=dim, batch_size=BATCH_SIZE)
    model = FlowMatchingMLP(data_dim=dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    step = 0
    while step < TRAIN_STEPS:
        for x in dataloader:
            if step >= TRAIN_STEPS:
                break

            x = x.float().to(DEVICE)
            batch_size = x.shape[0]

            # Sample t and epsilon
            t = torch.rand(batch_size, device=DEVICE).clamp(EPS, 1 - EPS)
            eps = torch.randn_like(x)

            # Construct zt = (1 - t) * x + t * eps
            t_expand = t.unsqueeze(-1)
            zt = (1 - t_expand) * x + t_expand * eps

            # Targets
            v_target = eps - x
            x_target = x

            # Forward: model outputs prediction
            output = model(zt, t)

            # Compute loss depending on pred_type and loss_type
            if pred_type == "v" and loss_type == "v":
                # model outputs v, loss in v-space
                loss = nn.functional.mse_loss(output, v_target)

            elif pred_type == "v" and loss_type == "x":
                # model outputs v, convert to x, loss in x-space
                x_pred = zt - t_expand * output
                loss = nn.functional.mse_loss(x_pred, x_target)

            elif pred_type == "x" and loss_type == "x":
                # model outputs x, loss in x-space
                loss = nn.functional.mse_loss(output, x_target)

            elif pred_type == "x" and loss_type == "v":
                # model outputs x, convert to v, loss in v-space
                v_pred = (zt - output) / t_expand
                loss = nn.functional.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5000 == 0:
                print(f"    Step {step:>6d}/{TRAIN_STEPS} | Loss: {loss.item():.6f}")

            step += 1

    print(f"    Done. Final loss: {loss.item():.6f}")

    # Save checkpoint
    ckpt_path = CKPT_DIR / f"{tag}.pt"
    torch.save(model.state_dict(), ckpt_path)

    return model


# ============================================================
# Euler ODE Sampling
# ============================================================
@torch.no_grad()
def euler_sample(model, num_samples: int, dim: int, pred_type: str):
    """
    Euler ODE sampling from t=1 to t=0.
    Handles both v-prediction and x-prediction models.
    """
    model.eval()
    z = torch.randn(num_samples, dim, device=DEVICE)
    dt = -1.0 / EULER_STEPS

    t_current = 1.0
    for _ in range(EULER_STEPS):
        t_batch = torch.full((num_samples,), t_current, device=DEVICE)
        output = model(z, t_batch)

        if pred_type == "v":
            # output is v directly
            v = output
        else:
            # output is x_pred, convert to v = (zt - x_pred) / t
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

    axes[1].scatter(generated_2d[:, 0], generated_2d[:, 1], s=1, alpha=0.5, color="orange")
    axes[1].set_title(title_right)
    axes[1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Running 36 experiments: 4 combinations × 3 datasets × 3 dims")
    print(f"{'='*60}")

    for pred_type, loss_type in COMBINATIONS:
        for dataset_name in DATASETS:
            for dim in DIMS:
                tag = f"{pred_type}_pred_{loss_type}_loss_{dataset_name}_D{dim}"

                # Train
                model = train(dataset_name, dim, pred_type, loss_type)

                # Generate samples
                samples = euler_sample(model, NUM_SAMPLES, dim, pred_type)
                samples_np = samples.cpu().numpy()

                # Load ground truth
                ds = ToyDiffusionDataset(name=dataset_name, dim=dim)

                # Project to 2D for visualization
                generated_2d = ds.to_2d(samples_np)
                real_2d = ds.to_2d(ds.data.numpy())

                # Plot
                save_path = OUTPUT_DIR / f"{tag}.png"
                plot_comparison(
                    real_2d=real_2d,
                    generated_2d=generated_2d,
                    title_left=f"{dataset_name} D={dim} — Ground Truth",
                    title_right=f"{dataset_name} D={dim} — {pred_type}-pred, {loss_type}-loss",
                    save_path=save_path,
                )
                print(f"    Saved: {save_path}")

    print(f"\n{'='*60}")
    print("All 36 experiments complete!")


if __name__ == "__main__":
    main()
