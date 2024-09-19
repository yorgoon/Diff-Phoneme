# config.py

import random
import numpy as np
import torch

# ========================================
# Random Seed for Reproducibility
# ========================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========================================
# Device Configuration
# ========================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========================================
# Data Paths and Directories
# ========================================

# Root directory for data files
# ROOT_DIR = "path/to/your/data/"  # Update this path accordingly
ROOT_DIR = "D:/13 class dataset/overt/08_30_2024_v1/" # Update this path accordingly
# ========================================
# Phoneme Vocabulary and Groups
# ========================================

PHONEME_VOCAB = {
    'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ay': 5,
    'b': 6, 'ch': 7, 'd': 8, 'dh': 9, 'eh': 10, 'er': 11,
    'ey': 12, 'f': 13, 'g': 14, 'hh': 15, 'ih': 16, 'iy': 17,
    'jh': 18, 'k': 19, 'l': 20, 'm': 21, 'n': 22, 'ng': 23,
    'ow': 24, 'oy': 25, 'p': 26, 'r': 27, 's': 28, 'sh': 29,
    't': 30, 'th': 31, 'uh': 32, 'uw': 33, 'v': 34, 'w': 35,
    'y': 36, 'z': 37, 'zh': 38, 'sil': 39  # Silence
}

PHONEME_GROUPS = {
    'vowels': ['iy', 'ih', 'eh', 'ae', 'aa', 'ao', 'uh', 'uw', 'ah', 'er'],
    'diphthongs': ['ey', 'ay', 'aw', 'oy', 'ow'],
    'stops_plosives': ['p', 'b', 't', 'd', 'k', 'g'],
    'affricates': ['ch', 'jh'],
    'fricatives': ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh'],
    'nasals': ['m', 'n', 'ng'],
    'liquids': ['l', 'r'],
    'semivowels': ['w', 'y'],
    'silence': ['sil']
}

# Create a reverse mapping from phoneme to group
PHONEME_TO_GROUP = {
    phoneme: group for group, phonemes in PHONEME_GROUPS.items() for phoneme in phonemes
}

# ========================================
# Word List and Phoneme Dictionary
# ========================================

WORD_LIST = [
    "ambulance",   # 0
    "clock",       # 1
    "hello",       # 2
    "help me",     # 3
    "light",       # 4
    "pain",        # 5
    "stop",        # 6
    "thank you",   # 7
    "toilet",      # 8
    "tv",          # 9
    "water",       # 10
    "yes",         # 11
    "resting state"  # 12
]

# Define phoneme sequences for each word
WORD_TO_PHONEME_DICT = {
    0: [PHONEME_VOCAB['ae'], PHONEME_VOCAB['m'], PHONEME_VOCAB['b'], PHONEME_VOCAB['y'], PHONEME_VOCAB['ah'],
        PHONEME_VOCAB['l'], PHONEME_VOCAB['ah'], PHONEME_VOCAB['n'], PHONEME_VOCAB['s'], PHONEME_VOCAB['sil']],  # "ambulance"
    1: [PHONEME_VOCAB['k'], PHONEME_VOCAB['l'], PHONEME_VOCAB['aa'], PHONEME_VOCAB['k'], PHONEME_VOCAB['sil']],  # "clock"
    2: [PHONEME_VOCAB['hh'], PHONEME_VOCAB['eh'], PHONEME_VOCAB['l'], PHONEME_VOCAB['ow'], PHONEME_VOCAB['sil']],  # "hello"
    3: [PHONEME_VOCAB['hh'], PHONEME_VOCAB['eh'], PHONEME_VOCAB['l'], PHONEME_VOCAB['p'],
        PHONEME_VOCAB['m'], PHONEME_VOCAB['iy'], PHONEME_VOCAB['sil']],  # "help me"
    4: [PHONEME_VOCAB['l'], PHONEME_VOCAB['ay'], PHONEME_VOCAB['t'], PHONEME_VOCAB['sil']],  # "light"
    5: [PHONEME_VOCAB['p'], PHONEME_VOCAB['ey'], PHONEME_VOCAB['n'], PHONEME_VOCAB['sil']],  # "pain"
    6: [PHONEME_VOCAB['s'], PHONEME_VOCAB['t'], PHONEME_VOCAB['aa'], PHONEME_VOCAB['p'], PHONEME_VOCAB['sil']],  # "stop"
    7: [PHONEME_VOCAB['th'], PHONEME_VOCAB['ae'], PHONEME_VOCAB['ng'], PHONEME_VOCAB['k'],
        PHONEME_VOCAB['y'], PHONEME_VOCAB['uw'], PHONEME_VOCAB['sil']],  # "thank you"
    8: [PHONEME_VOCAB['t'], PHONEME_VOCAB['oy'], PHONEME_VOCAB['l'], PHONEME_VOCAB['ah'],
        PHONEME_VOCAB['t'], PHONEME_VOCAB['sil']],  # "toilet"
    9: [PHONEME_VOCAB['t'], PHONEME_VOCAB['iy'], PHONEME_VOCAB['v'], PHONEME_VOCAB['iy'], PHONEME_VOCAB['sil']],  # "tv"
    10: [PHONEME_VOCAB['w'], PHONEME_VOCAB['ao'], PHONEME_VOCAB['t'], PHONEME_VOCAB['er'], PHONEME_VOCAB['sil']],  # "water"
    11: [PHONEME_VOCAB['y'], PHONEME_VOCAB['eh'], PHONEME_VOCAB['s'], PHONEME_VOCAB['sil']],  # "yes"
    12: [PHONEME_VOCAB['sil'], PHONEME_VOCAB['sil']]  # "resting state"
}

# ========================================
# Max Sequence Length
# ========================================

MAX_SEQ_LEN = max(len(seq) for seq in WORD_TO_PHONEME_DICT.values())

# ========================================
# Training Configuration
# ========================================

BATCH_SIZE = 32  # Adjust based on your GPU memory
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100

# ========================================
# Data Loading Configuration
# ========================================

# Subject and Session for Data Loading
SUBJECT = [4]
SESSION = 3

# Target Labels (if you want to select specific words)
TARGET_LABELS = list(range(len(WORD_LIST)))  # Use all words by default

# ========================================
# Diffusion Model Configuration (if applicable)
# ========================================

DIFFUSION_STEPS = 1000  # Number of diffusion steps
SCHEDULE = 'linear'     # Type of diffusion schedule

# ========================================
# Paths for Saved Models and Outputs
# ========================================

MODEL_SAVE_PATH = "models/"
OUTPUT_PATH = "outputs/"

# Ensure directories exist
import os
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
