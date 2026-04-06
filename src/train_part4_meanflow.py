"""
Part 4.2: MeanFlow — One-Step Generation
Train MeanFlow (x-prediction base) at D=32 on all 3 datasets.
Generate samples with 1, 2, 5 steps. Produce 9 figures.
"""

import torch
import torch.nn as nn
import torch.func
import matplotlib.pyplot as plt
from pathlib import Path
import time

from dataloader import ToyDiffusionDataset
from model import MeanFlowMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = ["swiss_roll", "gaussians", "circles"]
DATA_DIM = 32
BATCH_SIZE = 1024
LR = 1e-3
TRAIN_STEPS = 50000
NUM_SAMPLES = 2048
EPS = 1e-5
FM_RATIO = 0.5
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part4"
CKPT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints_part4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def train_meanflow(all_data, dataset_name):
    """Train MeanFlow model with 50/50 FM + MeanFlow consistency."""
    torch.manual_seed(SEED)
    model = MeanFlowMLP(data_dim=DATA_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_total = all_data.shape[0]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    model.train()
    t_start = time.time()

    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, n_total, (BATCH_SIZE,), device=DEVICE)
        x = all_data[idx]

        t = torch.rand(BATCH_SIZE, device=DEVICE).clamp(EPS, 1 - EPS)
        eps = torch.randn_like(x)
        t_exp = t.unsqueeze(-1)
        zt = (1 - t_exp) * x + t_exp * eps

        use_fm = torch.rand(1).item() < FM_RATIO

        if use_fm:
            h = torch.zeros(BATCH_SIZE, device=DEVICE)
            v_pred = model(zt, t, h)
            x_pred = zt - t_exp * v_pred
            loss = nn.functional.mse_loss(x_pred, x)
        else:
            r = torch.rand(BATCH_SIZE, device=DEVICE) * t
            h = (t - r).clamp(min=EPS)

            with torch.no_grad():
                v_instant = model(zt, t, torch.zeros(BATCH_SIZE, device=DEVICE))

            def fn(h_input):
                return model(zt, t, h_input)

            V_h, dV_dh = torch.func.jvp(fn, (h,), (torch.ones_like(h),))
            f_theta = V_h + h.unsqueeze(-1) * dV_dh
            loss = nn.functional.mse_loss(f_theta, v_instant.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10000 == 0 or step == TRAIN_STEPS - 1:
            elapsed = time.time() - t_start
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            mode = "FM" if use_fm else "MF"
            print(f"    Step {step:>6d}/{TRAIN_STEPS} | Loss: {loss.item():.6f} | {sps:.0f} steps/s [{mode}]")

    elapsed = time.time() - t_start
    print(f"    Done in {elapsed:.1f}s")

    ckpt_path = CKPT_DIR / f"meanflow_{dataset_name}_D{DATA_DIM}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"    Saved: {ckpt_path}")

    return model


@torch.no_grad()
def meanflow_sample(model, num_samples, dim, num_steps):
    """Sample using MeanFlow with N uniform steps from t=1 to t=0."""
    model.eval()
    z = torch.randn(num_samples, dim, device=DEVICE)
    h = 1.0 / num_steps
    t_current = 1.0

    for _ in range(num_steps):
        t_batch = torch.full((num_samples,), t_current, device=DEVICE)
        h_batch = torch.full((num_samples,), h, device=DEVICE)
        V = model(z, t_batch, h_batch)
        z = z - h * V
        t_current -= h

    return z


def plot_comparison(real_2d, generated_2d, title_left, title_right, save_path):
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


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Part 4.2: MeanFlow Training")
    print(f"Datasets: {DATASETS}, D={DATA_DIM}, Steps={TRAIN_STEPS}")
    print(f"{'=' * 60}")

    for ds_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        dataset = ToyDiffusionDataset(name=ds_name, dim=DATA_DIM)
        all_data = dataset.data.float().to(DEVICE)
        real_2d = dataset.to_2d(dataset.data.numpy())

        model = train_meanflow(all_data, ds_name)

        for n_steps in [1, 2, 5]:
            samples = meanflow_sample(model, NUM_SAMPLES, DATA_DIM, n_steps)
            samples_2d = dataset.to_2d(samples.cpu().numpy())

            save_path = OUTPUT_DIR / f"meanflow_{ds_name}_{n_steps}step.png"
            plot_comparison(
                real_2d=real_2d,
                generated_2d=samples_2d,
                title_left=f"{ds_name} D={DATA_DIM} — Ground Truth",
                title_right=f"{ds_name} D={DATA_DIM} — MeanFlow ({n_steps}-step)",
                save_path=save_path,
            )
            print(f"    Saved: {save_path}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("All MeanFlow experiments complete!")


if __name__ == "__main__":
    main()
