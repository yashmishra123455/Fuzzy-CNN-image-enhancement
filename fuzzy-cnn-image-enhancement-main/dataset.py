
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from config import Config
from feature_extractor import extract_features, features_to_vector

def compute_ideal_params(degraded: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Heuristically compute the brightness / contrast / gamma factors
    that would bring `degraded` closer to `reference`.

    Returns
    -------
    np.ndarray of shape (3,):
        [brightness_factor_norm, contrast_factor_norm, gamma_norm]
    All values normalised to [0, 1] for the CNN sigmoid output layer.
    """
    ref_mean = np.mean(reference) / 255.0 + 1e-8
    deg_mean = np.mean(degraded)  / 255.0 + 1e-8
    ref_std  = np.std(reference)  / 255.0 + 1e-8
    deg_std  = np.std(degraded)   / 255.0 + 1e-8

    brightness_factor = np.clip(ref_mean / deg_mean,
                                *Config.BRIGHTNESS_RANGE)
    contrast_factor   = np.clip(ref_std  / deg_std,
                                *Config.CONTRAST_RANGE)
    gamma = np.clip(np.log(0.5 + 1e-8) / np.log(deg_mean + 1e-8),
                    *Config.GAMMA_RANGE)

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo + 1e-8)

    return np.array([
        norm(brightness_factor, *Config.BRIGHTNESS_RANGE),
        norm(contrast_factor,   *Config.CONTRAST_RANGE),
        norm(gamma,             *Config.GAMMA_RANGE),
    ], dtype=np.float32)


class SyntheticEnhancementDataset(Dataset):

    def __init__(self, num_samples: int = 5000,
                 img_size: tuple = Config.IMAGE_SIZE):
        self.num_samples = num_samples
        self.img_size    = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        reference         = self._make_reference()
        degraded, mode    = self._apply_degradation(reference)
        feats             = extract_features(degraded)
        feat_vec          = features_to_vector(feats)
        targets           = compute_ideal_params(degraded, reference)

        return (torch.tensor(feat_vec, dtype=torch.float32),
                torch.tensor(targets,  dtype=torch.float32))

    def _make_reference(self) -> np.ndarray:
        H, W  = self.img_size
        base  = np.random.randint(60, 200, (H, W), dtype=np.uint8)
        noise = np.random.normal(0, 20, (H, W)).astype(np.float32)
        img   = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.5:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        return img

    def _apply_degradation(self, img: np.ndarray):
        mode = np.random.choice(["dark", "bright", "low_contrast", "noise"])

        if mode == "dark":
            f = np.random.uniform(0.3, 0.7)
            degraded = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)

        elif mode == "bright":
            f = np.random.uniform(1.4, 2.5)
            degraded = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)

        elif mode == "low_contrast":
            degraded = cv2.convertScaleAbs(img, alpha=0.4, beta=80)

        else:
            noise = np.random.normal(0, 30, img.shape).astype(np.float32)
            degraded = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return degraded, mode

