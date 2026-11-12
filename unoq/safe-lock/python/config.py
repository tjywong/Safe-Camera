"""Shared settings for every script in the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # data/<label>/<burst>_<n>.jpg
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fingernet.pt"
ONNX_PATH = MODEL_DIR / "fingernet.onnx"
CAMERA_FILE = ROOT / "camera.json"  # which camera index is the USB webcam

# Classes: 0..5 fingers. "0" means a fist OR no hand at all -- the lock uses
# it as the "hand removed" signal between digits.
NUM_CLASSES = 6

SAVE_SIZE = 128   # size images are stored at on disk (room for crop augmentation)
IMG_SIZE = 64     # size the network actually sees

# The secret: three numbers, each 1-5, entered in order.
COMBINATION = (3, 1, 4)

# Lock behaviour
CONFIDENCE = 0.80      # softmax probability needed to trust a prediction
STABLE_FRAMES = 8      # same answer this many frames in a row = "entered"
ENTRY_TIMEOUT = 20.0   # seconds to enter all three digits after the button
UNLOCK_SECONDS = 5.0   # how long the lock stays open
