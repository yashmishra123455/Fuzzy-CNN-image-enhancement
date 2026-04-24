
import cv2, numpy as np
from scipy.stats import entropy as scipy_entropy

def extract_features(image):
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
        if img.max() > 1.0: img /= 255.0
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(img))
    std_val  = float(np.std(img))
    hist, _  = np.histogram(img, bins=256, range=(0.0, 1.0))
    hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-8)
    entropy_val  = float(scipy_entropy(hist_norm + 1e-8))
    dark_ratio   = float(np.mean(img < 0.3))
    bright_ratio = float(np.mean(img > 0.7))
    return {"mean": mean_val, "std": std_val, "entropy": entropy_val,
            "hist": hist_norm, "dark_ratio": dark_ratio, "bright_ratio": bright_ratio}

def features_to_vector(features):
    scalars = np.array([features["mean"], features["std"],
                        features["entropy"], features["dark_ratio"]], dtype=np.float32)
    return np.concatenate([scalars, features["hist"]])
