
import torch
import torch.nn as nn
import numpy as np
from config import Config

class EnhancementCNN(nn.Module):
    

    def __init__(self,
                 input_dim:  int   = 260,
                 output_dim: int   = 3,
                 dropout:    float = Config.DROPOUT_RATE):
        super().__init__()

        self.network = nn.Sequential(
            #Block 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            #Block 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            #Block 3 
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            #Output
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.network(x)

    def predict_params(self, x: torch.Tensor) -> dict:
        
        self.eval()
        with torch.no_grad():
            raw = self.forward(x)

        def rescale(v, lo, hi):
            return lo + v * (hi - lo)

        raw_np = raw.cpu().numpy()
        return {
            "brightness_factor": rescale(raw_np[:, 0], *Config.BRIGHTNESS_RANGE),
            "contrast_factor":   rescale(raw_np[:, 1], *Config.CONTRAST_RANGE),
            "gamma":             rescale(raw_np[:, 2], *Config.GAMMA_RANGE),
        }
