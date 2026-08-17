
class Config:
    IMAGE_SIZE           = (128, 128)
    NORMALIZE_MEAN       = [0.5]
    NORMALIZE_STD        = [0.5]
    BATCH_SIZE           = 32
    LEARNING_RATE        = 1e-3
    NUM_EPOCHS           = 30
    WEIGHT_DECAY         = 1e-4
    DROPOUT_RATE         = 0.3
    BRIGHTNESS_RANGE     = (0.5, 2.0)
    CONTRAST_RANGE       = (0.5, 2.0)
    GAMMA_RANGE          = (0.4, 2.5)
    BRIGHTNESS_UNIVERSE  = (0.0, 1.0)
    CONTRAST_UNIVERSE    = (0.0, 1.0)
    ADJUSTMENT_UNIVERSE  = (-1.0, 1.0)
    MODEL_SAVE_PATH      = "cnn_enhancer.pth"
    SAMPLE_IMAGE         = "sample.jpg"
