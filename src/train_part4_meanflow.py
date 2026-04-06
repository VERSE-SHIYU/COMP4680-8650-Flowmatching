"""
Part 4.2: MeanFlow — One-Step Generation
Train MeanFlow (x-prediction base) at D=32; generate 1/2/5-step samples (9 figures).

Stable training stack:
  - Warmup: pure FM (h=0); LR fixed at LR (no cosine during this phase).
  - After warmup: cosine LR decay; 50/50 FM vs MF; EMA teacher for MF target.
  - MF: Beta(2,5) on h in [0, h_max]; h_max cosine curriculum 0.05 -> 1.0.
  - Gradient clipping; CSV logs (FM vs MF loss per 1k steps).
  - Checkpoints store model + EMA; sampling defaults to EMA weights.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataloader import ToyDiffusionDataset
from model import MeanFlowMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIM = 32
BATCH_SIZE = 1024
LR = 1e-3
ETA_MIN = 1e-5
TRAIN_STEPS = 400_000
WARMUP_STEPS = 40_000
NUM_SAMPLES = 2048
EPS = 1e-5
FM_RATIO = 0.5
SEED = 42

EMA_DECAY = 0.999
GRAD_CLIP_NORM = 1.0
BETA_A = 2.0
BETA_B = 5.0
LOG_EVERY = 1_000
PRINT_EVERY = 10_000

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part4"
CKPT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints_part4"
LOG_DIR = OUTPUT_DIR / "logs"


def parse_args():
    p = argparse.ArgumentParser(description="Part 4.2 MeanFlow training")
    p.add_argument(
        "--datasets",
        type=str,
        default="swiss_roll,gaussians,circles",
        help="Comma-separated dataset names",
    )
    p.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    p.add_argument("--no-ema", action="store_true", help="Train without EMA target; sample with student only")
    p.add_argument(
        "--sample-raw",
        action="store_true",
        help="Use student weights for sampling instead of EMA (only if EMA enabled)",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def set_learning_rate(optimizer: torch.optim.Optimizer, step: int, train_steps: int, warmup_steps: int) -> float:
    """Constant LR during warmup; cosine decay LR -> ETA_MIN over the rest."""
    if step < warmup_steps:
        lr = LR
    else:
        t = (step - warmup_steps) / max(train_steps - warmup_steps, 1)
        t = min(max(t, 0.0), 1.0)
        lr = ETA_MIN + (LR - ETA_MIN) * 0.5 * (1.0 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
        p_ema.mul_(decay).add_(p, alpha=1.0 - decay)


def h_max_curriculum(step: int, train_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return 0.0
    denom = train_steps - warmup_steps
    mf_progress = (step - warmup_steps) / max(denom, 1)
    mf_progress = min(max(mf_progress, 0.0), 1.0)
    return float(0.05 + 0.95 * 0.5 * (1.0 - math.cos(math.pi * mf_progress)))


def train_meanflow(
    all_data: torch.Tensor,
    dataset_name: str,
    train_steps: int,
    warmup_steps: int,
    use_ema: bool,
    seed: int,
):
    torch.manual_seed(seed)
    model = MeanFlowMLP(data_dim=DATA_DIM).to(DEVICE)
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_total = all_data.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params | EMA target: {use_ema}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / f"meanflow_{dataset_name}_train.csv"
    log_f = open(csv_path, "w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(
        ["step", "lr", "h_max", "loss_fm_avg_1k", "loss_mf_avg_1k", "fm_batches_1k", "mf_batches_1k"]
    )

    model.train()
    t_start = time.time()

    log_fm_sum = 0.0
    log_mf_sum = 0.0
    log_fm_n = 0
    log_mf_n = 0

    win_fm_sum = 0.0
    win_mf_sum = 0.0
    win_fm_n = 0
    win_mf_n = 0

    for step in range(train_steps):
        lr_now = set_learning_rate(optimizer, step, train_steps, warmup_steps)
        h_max = h_max_curriculum(step, train_steps, warmup_steps)

        idx = torch.randint(0, n_total, (BATCH_SIZE,), device=DEVICE)
        x = all_data[idx]

        t = torch.rand(BATCH_SIZE, device=DEVICE).clamp(EPS, 1 - EPS)
        eps = torch.randn_like(x)
        t_exp = t.unsqueeze(-1)
        zt = (1 - t_exp) * x + t_exp * eps

        use_fm = (step < warmup_steps) or (torch.rand(1).item() < FM_RATIO)

        if use_fm:
            h = torch.zeros(BATCH_SIZE, device=DEVICE)
            v_pred = model(zt, t, h)
            x_pred = zt - t_exp * v_pred
            loss = nn.functional.mse_loss(x_pred, x)
        else:
            u = torch.distributions.Beta(
                torch.tensor(BETA_A, device=DEVICE), torch.tensor(BETA_B, device=DEVICE)
            ).sample((BATCH_SIZE,))
            h = u * h_max
            h = torch.min(h, t - EPS).clamp(min=EPS)

            def fn(h_input):
                return model(zt, t, h_input)

            V_h, dV_dh = torch.func.jvp(fn, (h,), (torch.ones_like(h),))
            f_theta = V_h + h.unsqueeze(-1) * dV_dh

            with torch.no_grad():
                z_reached = zt - h.unsqueeze(-1) * V_h.detach()
                t_reached = (t - h).clamp(min=EPS)
                h_zero = torch.zeros(BATCH_SIZE, device=DEVICE)
                if use_ema and ema_model is not None:
                    target = ema_model(z_reached, t_reached, h_zero)
                else:
                    target = model(z_reached, t_reached, h_zero)

            loss = nn.functional.mse_loss(f_theta, target.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        if use_ema and ema_model is not None:
            update_ema(model, ema_model, EMA_DECAY)

        if use_fm:
            log_fm_sum += loss.item()
            log_fm_n += 1
            win_fm_sum += loss.item()
            win_fm_n += 1
        else:
            log_mf_sum += loss.item()
            log_mf_n += 1
            win_mf_sum += loss.item()
            win_mf_n += 1

        if (step + 1) % LOG_EVERY == 0:
            fm_a = log_fm_sum / max(log_fm_n, 1)
            mf_a = log_mf_sum / max(log_mf_n, 1)
            log_writer.writerow(
                [step + 1, f"{lr_now:.8f}", f"{h_max:.6f}", fm_a, mf_a, log_fm_n, log_mf_n]
            )
            log_f.flush()
            log_fm_sum = 0.0
            log_mf_sum = 0.0
            log_fm_n = 0
            log_mf_n = 0

        if step % PRINT_EVERY == 0 or step == train_steps - 1:
            elapsed = time.time() - t_start
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            mode = "FM" if use_fm else "MF"
            fm_win = win_fm_sum / max(win_fm_n, 1)
            mf_win = win_mf_sum / max(win_mf_n, 1)
            print(
                f"    Step {step:>6d}/{train_steps} | loss={loss.item():.6f} | "
                f"lr={lr_now:.2e} | FM_avg={fm_win:.6f} MF_avg={mf_win:.6f} | "
                f"{sps:.0f} steps/s [{mode}] h_max={h_max:.3f}"
            )
            win_fm_sum = 0.0
            win_mf_sum = 0.0
            win_fm_n = 0
            win_mf_n = 0

    log_f.close()
    elapsed = time.time() - t_start
    print(f"    Done in {elapsed:.1f}s | log: {csv_path}")

    ckpt_path = CKPT_DIR / f"meanflow_{dataset_name}_D{DATA_DIM}.pt"
    if use_ema and ema_model is not None:
        torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(), "meta": {"use_ema": True}}, ckpt_path)
    else:
        torch.save({"model": model.state_dict(), "meta": {"use_ema": False}}, ckpt_path)
    print(f"    Saved: {ckpt_path}")

    return model, ema_model


@torch.no_grad()
def meanflow_sample(net: nn.Module, num_samples: int, dim: int, num_steps: int):
    """Sample with N uniform steps from t=1 to t=0."""
    net.eval()
    z = torch.randn(num_samples, dim, device=DEVICE)
    h = 1.0 / num_steps
    t_current = 1.0

    for _ in range(num_steps):
        t_batch = torch.full((num_samples,), t_current, device=DEVICE)
        h_batch = torch.full((num_samples,), h, device=DEVICE)
        V = net(z, t_batch, h_batch)
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
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    train_steps = args.train_steps
    warmup_steps = args.warmup
    use_ema = not args.no_ema
    sample_with_ema = use_ema and not args.sample_raw

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("Part 4.2: MeanFlow Training (stable stack)")
    print(f"Datasets: {datasets}, D={DATA_DIM}, steps={train_steps}, warmup={warmup_steps}")
    print(
        f"LR: fixed {LR} during warmup, then cosine -> {ETA_MIN}; "
        f"grad_clip={GRAD_CLIP_NORM}; Beta({BETA_A},{BETA_B}) for h; EMA train={use_ema}; sample={'EMA' if sample_with_ema else 'student'}"
    )
    print("=" * 60)

    for ds_name in datasets:
        print(f"\n{'=' * 60}\nDataset: {ds_name}\n{'=' * 60}")

        dataset = ToyDiffusionDataset(name=ds_name, dim=DATA_DIM)
        all_data = dataset.data.float().to(DEVICE)
        real_2d = dataset.to_2d(dataset.data.numpy())

        model, ema_model = train_meanflow(
            all_data,
            ds_name,
            train_steps=train_steps,
            warmup_steps=warmup_steps,
            use_ema=use_ema,
            seed=args.seed,
        )

        sampler = ema_model if sample_with_ema else model
        tag = "EMA" if sample_with_ema else "student"

        for n_steps in [1, 2, 5]:
            samples = meanflow_sample(sampler, NUM_SAMPLES, DATA_DIM, n_steps)
            samples_2d = dataset.to_2d(samples.cpu().numpy())

            save_path = OUTPUT_DIR / f"meanflow_{ds_name}_{n_steps}step.png"
            plot_comparison(
                real_2d=real_2d,
                generated_2d=samples_2d,
                title_left=f"{ds_name} D={DATA_DIM} — Ground Truth",
                title_right=f"{ds_name} D={DATA_DIM} — MeanFlow ({n_steps}-step, {tag})",
                save_path=save_path,
            )
            print(f"    Saved: {save_path}")

        del model
        if ema_model is not None:
            del ema_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\nAll MeanFlow experiments complete!")


if __name__ == "__main__":
    main()
