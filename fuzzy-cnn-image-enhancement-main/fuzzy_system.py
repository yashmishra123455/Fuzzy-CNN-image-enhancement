
import numpy as np
from config import Config


# ── Membership Functions ─────────────────────────────

def triangular_mf(x, a, b, c):
    return max(min((x - a) / (b - a + 1e-6),
                   (c - x) / (c - b + 1e-6)), 0.0)


def trapezoidal_mf(x, a, b, c, d):
    return max(min((x - a) / (b - a + 1e-6),
                   1.0,
                   (d - x) / (d - c + 1e-6)), 0.0)


def brightness_mf(brightness: float) -> dict:
    return {
        "dark":   trapezoidal_mf(brightness, 0.0, 0.0, 0.25, 0.45),
        "medium": triangular_mf(brightness, 0.3, 0.5, 0.7),
        "bright": trapezoidal_mf(brightness, 0.55, 0.75, 1.0, 1.0),
    }


def contrast_mf(contrast: float) -> dict:
    return {
        "low":    trapezoidal_mf(contrast, 0.0, 0.0, 0.2, 0.4),
        "medium": triangular_mf(contrast, 0.3, 0.5, 0.7),
        "high":   trapezoidal_mf(contrast, 0.6, 0.8, 1.0, 1.0),
    }


def adjustment_mf_vec(x_vals: np.ndarray, label: str) -> np.ndarray:
    params = {
        "decrease":  (-1.0, -0.5, 0.0),
        "no_change": (-0.2,  0.0, 0.2),
        "increase":  ( 0.0,  0.5, 1.0),
    }
    a, b, c = params[label]
    return np.array([triangular_mf(x, a, b, c) for x in x_vals], dtype=np.float32)


RULE_BASE = [
    ("dark",   "low",    "increase",  1.0),
    ("dark",   "medium", "increase",  0.9),
    ("dark",   "high",   "increase",  0.7),
    ("medium", "low",    "increase",  0.6),
    ("medium", "medium", "no_change", 1.0),
    ("medium", "high",   "decrease",  0.5),
    ("bright", "low",    "no_change", 0.5),
    ("bright", "medium", "decrease",  0.9),
    ("bright", "high",   "decrease",  1.0),   
]


class FuzzyEnhancementSystem:

    def __init__(self, resolution: int = 500):
        lo, hi     = Config.ADJUSTMENT_UNIVERSE
        self.x_out = np.linspace(lo, hi, resolution)

    def infer(self, brightness: float, contrast: float) -> dict:

        b_mems = brightness_mf(brightness)
        c_mems = contrast_mf(contrast)

        aggregated   = np.zeros_like(self.x_out)
        active_rules = []

        for (b_lbl, c_lbl, out_lbl, weight) in RULE_BASE:

            strength = min(b_mems[b_lbl], c_mems[c_lbl]) * weight

            if strength < 1e-4:
                continue

            out_mf  = adjustment_mf_vec(self.x_out, out_lbl)
            clipped = np.minimum(out_mf, strength)

            aggregated = np.maximum(aggregated, clipped)

            active_rules.append({
                "rule": "IF {b_lbl} AND {c_lbl} -> {out_lbl}".format(
                    b_lbl=b_lbl, c_lbl=c_lbl, out_lbl=out_lbl),
                "strength": round(strength, 4),
            })

        denom = np.sum(aggregated)
        if denom < 1e-8:
            crisp = 0.0
        else:
            crisp = float(np.sum(self.x_out * aggregated) / denom)

        return {
            "adjustment":   crisp,
            "active_rules": active_rules,
            "aggregated":   aggregated,
        }

    def get_universe(self):
        return self.x_out
