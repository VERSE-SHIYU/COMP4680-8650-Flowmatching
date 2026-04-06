"""
Part 1.1: Data Visualization
For each dataset (swiss_roll, gaussians, circles):
  - Plot original 2D data
  - Plot 32D data projected back to 2D via to_2d()
Total: 6 figures
"""

import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import ToyDiffusionDataset

DATASETS = ["swiss_roll", "gaussians", "circles"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part1_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_scatter(samples_2d, title, save_path):
    """Draw a scatter plot and save to file."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    for name in DATASETS:
        # Original 2D data
        ds_2d = ToyDiffusionDataset(name=name, dim=2)
        samples_2d = ds_2d.data.numpy()
        plot_scatter(
            samples_2d,
            title=f"{name} — original 2D",
            save_path=OUTPUT_DIR / f"{name}_2d.png",
        )

        # 32D data projected back to 2D
        ds_32d = ToyDiffusionDataset(name=name, dim=32)
        samples_32d = ds_32d.data.numpy()
        samples_proj = ds_32d.to_2d(samples_32d)
        plot_scatter(
            samples_proj,
            title=f"{name} — 32D projected back to 2D",
            save_path=OUTPUT_DIR / f"{name}_32d_proj.png",
        )


if __name__ == "__main__":
    main()
