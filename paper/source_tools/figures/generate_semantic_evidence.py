from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.stats import spearmanr

from models.resnet_cifar import resnet56


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "experiments" / "baseline_cifar100_resnet56.pth"
DATA_ROOT = ROOT / "data"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device) -> torch.nn.Module:
    model = resnet56(num_classes=100).to(device)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_batch(batch_size: int = 128):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    ds = torchvision.datasets.CIFAR100(root=str(DATA_ROOT), train=False, download=False, transform=tf)
    ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return next(iter(ld))


def compute_gra_semantic_scores(feat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, rho: float = 0.5) -> np.ndarray:
    # feat: [B, C, H, W], logits: [B, K], labels: [B]
    a = feat.mean(dim=(2, 3)).detach().cpu().numpy()  # [B, C]
    z = logits.gather(1, labels.view(-1, 1)).squeeze(1).detach().cpu().numpy()  # [B]

    diff = np.abs(a - z[:, None])  # [B, C]
    d_min = float(diff.min())
    d_max = float(diff.max())
    xi = (d_min + rho * d_max) / (diff + rho * d_max + 1e-12)
    gamma = xi.mean(axis=0)
    return gamma


def channel_l1_scores(layer: torch.nn.Conv2d) -> np.ndarray:
    w = layer.weight.detach().cpu().numpy()  # [C_out, C_in, k, k]
    return np.mean(np.abs(w), axis=(1, 2, 3))


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)
    images, labels = load_batch(batch_size=128)
    images = images.to(device)
    labels = labels.to(device)

    target_layer = model.layer3[-1].conv2
    feat_holder = {}

    def hook_fn(_m, _i, o):
        feat_holder["feat"] = o.detach()

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        logits = model(images)
    handle.remove()

    feat = feat_holder["feat"]

    gra = compute_gra_semantic_scores(feat, logits, labels)
    l1 = channel_l1_scores(target_layer)

    gra_n = minmax(gra)
    l1_n = minmax(l1)

    rho_sp, _ = spearmanr(gra_n, l1_n)

    # disagreement channels
    score_sem = gra_n - l1_n
    score_struct = l1_n - gra_n
    sem_idx = np.argsort(score_sem)[-2:][::-1]
    struct_idx = np.argsort(score_struct)[-2:][::-1]

    sample_id = 0
    img = images[sample_id].detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.2675, 0.2565, 0.2761]) + np.array([0.5071, 0.4867, 0.4408])
    img = np.clip(img, 0.0, 1.0)

    fmap = feat[sample_id].detach().cpu().numpy()

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.facecolor": "#F9FAFB",
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=(10.8, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1.0, 1.0, 0.06], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.16)

    # scatter
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.scatter(l1_n, gra_n, s=18, alpha=0.70, c="#6B7280", edgecolors="none", label="channels")
    ax0.scatter(l1_n[sem_idx], gra_n[sem_idx], s=52, c="#0B5FA5", marker="o", label="semantic-priority")
    ax0.scatter(l1_n[struct_idx], gra_n[struct_idx], s=52, c="#F28E2B", marker="s", label="structural-priority")
    ax0.plot([0, 1], [0, 1], "--", lw=1.0, color="#9CA3AF")
    ax0.set_xlim(-0.02, 1.02)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_xlabel("Normalized L1-Norm score")
    ax0.set_ylabel("Normalized GRA semantic score")
    ax0.set_title(f"(a) Channel-ranking disagreement\nSpearman $r$ = {rho_sp:.3f}", loc="left")
    ax0.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax0.legend(loc="lower right", frameon=False, fontsize=8)

    # input
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img)
    ax1.set_title("(b) Input sample")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(img)
    ax2.set_title("Input sample")
    ax2.axis("off")

    # semantic-priority maps
    ax3 = fig.add_subplot(gs[1, 1])
    fm_sem = fmap[sem_idx[0]]
    im_sem = ax3.imshow(fm_sem, cmap="magma")
    ax3.set_title(
        f"(c) Semantic-priority channel {int(sem_idx[0])}\nGRA={gra_n[sem_idx[0]]:.2f}, L1={l1_n[sem_idx[0]]:.2f}"
    )
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 2])
    fm_struct = fmap[struct_idx[0]]
    im_struct = ax4.imshow(fm_struct, cmap="viridis")
    ax4.set_title(
        f"(d) Structural-priority channel {int(struct_idx[0])}\nGRA={gra_n[struct_idx[0]]:.2f}, L1={l1_n[struct_idx[0]]:.2f}"
    )
    ax4.axis("off")

    cax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(im_sem, cax=cax)
    cb.set_label("Activation intensity", fontsize=9)

    fig.subplots_adjust(left=0.055, right=0.97, top=0.93, bottom=0.08)

    out_pdf = OUT_DIR / "fig_semantic_evidence.pdf"
    out_png = OUT_DIR / "fig_semantic_evidence.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, dpi=700)
    plt.close(fig)

    report = OUT_DIR / "fig_semantic_evidence_stats.txt"
    report.write_text(
        "\n".join(
            [
                f"spearman_r={rho_sp:.6f}",
                f"semantic_priority_channels={sem_idx.tolist()}",
                f"structural_priority_channels={struct_idx.tolist()}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Stats: {report}")


if __name__ == "__main__":
    main()
