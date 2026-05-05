import os

# --- PATH CONFIGUREATIONS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "clean_dataset")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models")

# esp32 path to save
ESP32_MAIN_DIR = os.path.join(PROJECT_ROOT, "esp32", "main") 

# --- DATASET SETTINGS ---
CLASSES = ["0", "2", "3"]
NUM_CLASSES = len(CLASSES)
SPLITS = ["train", "validation", "test"]
FOLDERS_TO_CLEAR = ["0","2", "3"]


# --- TRAINING HYPERPARAMETERS ---
IMG_WIDTH = 160
IMG_HEIGHT = 120
BATCH_SIZE = 32
EPOCHS = 50

# Set manually 
MANUAL_CLASS_WEIGHTS = {
    0: 1.0,   # user face images
    1: 1.3,   # unknown faces
    2: 1.0,   # background
}