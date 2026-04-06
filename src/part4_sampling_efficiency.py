"""
Part 4.1: Sampling Efficiency
Evaluate x-pred + x-loss (best from Part 2) at D=32 on swiss_roll
with Euler step counts: 1, 2, 5, 10, 20, 50, 100, 200.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataloader import ToyDiffusionDataset
from model import FlowMatchingMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIM = 32
NUM_SAMPLES = 2048
EPS = 1e-5
STEP_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200]

CKPT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints_part2"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def euler_sample_x_pred(model, num_samples, dim, num_steps):
    model.eval()
    z = torch.randn(num_samples, dim, device=DEVICE)
    dt = -1.0 / num_steps
    t_current = 1.0

    for _ in range(num_steps):
        t_batch = torch.full((num_samples,), t_current, device=DEVICE)
        x_pred = model(z, t_batch)
        t_clamped = max(t_current, EPS)
        v = (z - x_pred) / t_clamped
        z = z + v * dt
        t_current += dt

    return z


def main():
    print(f"Device: {DEVICE}")
    print(f"Part 4.1: Sampling Efficiency Experiment")
    print(f"{'=' * 60}")

    datasets = ["swiss_roll", "gaussians", "circles"]

    for ds_name in datasets:
        ckpt_path = CKPT_DIR / f"x_pred_x_loss_{ds_name}_D{DATA_DIM}.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, skipping")
            continue

        model = FlowMatchingMLP(data_dim=DATA_DIM).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()

        dataset = ToyDiffusionDataset(name=ds_name, dim=DATA_DIM)
        real_2d = dataset.to_2d(dataset.data.numpy())

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()

        for i, n_steps in enumerate(STEP_COUNTS):
            samples = euler_sample_x_pred(model, NUM_SAMPLES, DATA_DIM, n_steps)
            samples_2d = dataset.to_2d(samples.cpu().numpy())

            axes[i].scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, color="orange")
            axes[i].set_title(f"{n_steps} steps", fontsize=14)
            axes[i].set_aspect("equal")

        fig.suptitle(f"Sampling Efficiency: x-pred + x-loss, {ds_name} D={DATA_DIM}", fontsize=16, fontweight="bold")
        fig.tight_layout()
        save_path = OUTPUT_DIR / f"sampling_efficiency_{ds_name}_D{DATA_DIM}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

        del model
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
