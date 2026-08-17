
import torch
from cnn_model import EnhancementCNN
from config import Config

def load_model(path=Config.MODEL_SAVE_PATH):
    """Load a saved model from disk."""
    model = EnhancementCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
