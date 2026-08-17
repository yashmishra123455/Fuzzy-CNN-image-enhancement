
import cv2, numpy as np, torch
from config import Config
from feature_extractor import extract_features, features_to_vector
from cnn_model import EnhancementCNN
from fuzzy_system import FuzzyEnhancementSystem

def apply_enhancement(image: np.ndarray,
                      brightness_factor: float,
                      contrast_factor:   float,
                      gamma:             float) -> np.ndarray:
    
    # Alpha-beta transform 
    beta     = int((brightness_factor - 1.0) * 50)   # [-50, +50] pixel shift
    alpha    = contrast_factor
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    #Gamma correction via LUT (fast: precompute once for all 256 values) 
    inv_gamma = 1.0 / max(gamma, 0.1)
    lut = np.array([
        np.clip(((i / 255.0) ** inv_gamma) * 255, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)

    return cv2.LUT(adjusted, lut)


class ImageEnhancer:
    

    def __init__(self, model_path: str = Config.MODEL_SAVE_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

       
        self.cnn = EnhancementCNN()
        self.cnn.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.cnn.eval()
        self.cnn.to(self.device)

        
        self.fis = FuzzyEnhancementSystem()

        print(f"[Enhancer] CNN loaded from '{model_path}'")
        print(f"[Enhancer] Fuzzy system initialised.")

    

    def enhance(self, image: np.ndarray) -> tuple:
        
        
        gray  = self._to_gray(image)
        feats = extract_features(gray)
        vec   = features_to_vector(feats)

        
        x_tensor   = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        raw_params = self.cnn.predict_params(x_tensor)

        bf = float(raw_params["brightness_factor"][0])
        cf = float(raw_params["contrast_factor"][0])
        g  = float(raw_params["gamma"][0])

        
        norm_b = np.clip(feats["mean"] / 255.0, 0.0, 1.0)
        norm_c = np.clip(feats["std"] / 255.0, 0.0, 1.0)

        fuzzy_result = self.fis.infer(norm_b, norm_c)
        adjustment   = fuzzy_result["adjustment"]      # in [-1, 1]

    
        fuzzy_scale = 1.0 + 0.5 * adjustment
        bf = np.clip(bf * fuzzy_scale, *Config.BRIGHTNESS_RANGE)
        cf = np.clip(cf * fuzzy_scale, *Config.CONTRAST_RANGE)

        
        enhanced = apply_enhancement(image, bf, cf, g)

        info = {
            "features":          feats,
            "brightness_factor": bf,
            "contrast_factor":   cf,
            "gamma":             g,
            "fuzzy_adjustment":  adjustment,
            "fuzzy_scale":       fuzzy_scale,
            "active_rules":      fuzzy_result["active_rules"],
        }

        return enhanced, info

    def enhance_fuzzy_only(self, image: np.ndarray) -> tuple:
        
        gray  = self._to_gray(image)
        feats = extract_features(gray)

        fuzzy_result = self.fis.infer(feats["mean"], feats["std"])
        adjustment   = fuzzy_result["adjustment"]
        fuzzy_scale  = 1.0 + 0.5 * adjustment

        enhanced = apply_enhancement(image,
                                     brightness_factor=1.0 * fuzzy_scale,
                                     contrast_factor=1.0   * fuzzy_scale,
                                     gamma=1.0)
        return enhanced, fuzzy_result

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
