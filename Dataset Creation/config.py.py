import torch

BASE_DIR = r"D:\arun work\melt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 50
FREEZE_EPOCHS = 3
LR = 1e-4
LATENT_DIM = 256
FIXED_AUDIO_LEN = 80000
MODALITY_DROPOUT_P = 0.2