
import cv2, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fuzzy_system import FuzzyEnhancementSystem, adjustment_mf_vec
from config import Config

def show_enhancement_results(original, fuzzy, cnn_fuzzy, info, metrics,
                              save_path="results.png"):
    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)
    images      = [original, fuzzy, cnn_fuzzy]
    titles      = ["Original", "Fuzzy only", "CNN + Fuzzy"]
    hist_colors = ["steelblue", "coral", "seagreen"]
    metric_lbls = ["",
        f"PSNR={metrics["psnr_fuzzy"]:.1f} dB  SSIM={metrics["ssim_fuzzy"]:.3f}",
        f"PSNR={metrics["psnr_cnn"]:.1f} dB  SSIM={metrics["ssim_cnn"]:.3f}"]
    for col, (img, title, color, mlbl) in enumerate(zip(images, titles, hist_colors, metric_lbls)):
        ax_img = fig.add_subplot(gs[0, col])
        disp   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        ax_img.imshow(disp, cmap="gray" if img.ndim == 2 else None)
        ax_img.set_title(f"{title}\n{mlbl}", fontsize=9)
        ax_img.axis("off")
        ax_hist = fig.add_subplot(gs[1, col])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        ax_hist.hist(gray.ravel(), bins=64, range=(0,255), color=color, alpha=0.85, edgecolor="none")
        ax_hist.set_xlabel("Pixel intensity", fontsize=9)
        ax_hist.set_ylabel("Count", fontsize=9)
        ax_hist.set_title("Histogram", fontsize=9)
    fig.suptitle("Fuzzy Logic Image Enhancement — Tuned by CNN", fontsize=13, y=1.01)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
