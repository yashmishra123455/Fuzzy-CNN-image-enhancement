
import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(psnr_fn(img1, img2, data_range=255))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(ssim_fn(img1, img2, data_range=255))


def evaluate_enhancement(original:   np.ndarray,
                          fuzzy_only: np.ndarray,
                          cnn_fuzzy:  np.ndarray) -> dict:
    
    psnr_fuzzy = compute_psnr(original, fuzzy_only)
    psnr_cnn   = compute_psnr(original, cnn_fuzzy)
    ssim_fuzzy = compute_ssim(original, fuzzy_only)
    ssim_cnn   = compute_ssim(original, cnn_fuzzy)

    winner_psnr = "CNN+Fuzzy" if psnr_cnn > psnr_fuzzy else "Fuzzy only"
    winner_ssim = "CNN+Fuzzy" if ssim_cnn > ssim_fuzzy else "Fuzzy only"

    report = (
        
        "IMAGE ENHANCEMENT EVALUATION"
        f" Method │  PSNR (dB) │ SSIM"
        f"Fuzzy only{psnr_fuzzy:8.3f} | {ssim_fuzzy:.5f}"
        f"CNN + Fuzzy{psnr_cnn:8.3f} | {ssim_cnn:.5f}"
        f"Best PSNR: {winner_psnr:<10s}"
        f"Best SSIM: {winner_ssim:<10s}"
    )
    print(report)

    return {
        "psnr_fuzzy": psnr_fuzzy,
        "psnr_cnn":   psnr_cnn,
        "ssim_fuzzy": ssim_fuzzy,
        "ssim_cnn":   ssim_cnn,
        "report":     report,
    }

