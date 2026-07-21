import os

# ── Paths ────────────────────────────────────────────────────────────────────
# All paths are relative to the project root so the repo works out of the box
# after cloning. Place the dataset zip at data/archive.zip before running.
_ROOT       = os.path.dirname(os.path.abspath(__file__))
DATASET_ZIP = os.path.join(_ROOT, "data", "archive.zip")
DATASET_DIR = os.path.join(_ROOT, "data")
OUTPUT_DIR  = os.path.join(_ROOT, "outputs")

# image size for MobileNetV3
IMG_SIZE = 224
USE_BACKGROUND_REMOVED = True

# training settings
BATCH_SIZE   = 16
NUM_EPOCHS   = 30
LR           = 1e-3
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4
NUM_WORKERS  = 0   # must be 0 on Windows
SEED         = 42
LOG_INTERVAL = 10

# dataset split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# focal loss gamma parameter
FOCAL_GAMMA = 2.0

# hazard weights (h_c) for each class based on WHO biomedical waste guidelines
# higher value = model penalized more for missing that class
HAZARD_WEIGHTS = {
    # sharps - highest risk (direct injury + bloodborne pathogens)
    "used_syringes":           3.0,
    "scalpels":                3.0,
    "episiotomy_scissors":     3.0,
    "mayo_scissors":           3.0,
    "stitch_removal_scissors": 3.0,
    "tweezers":                2.8,
    "forceps":                 2.8,
    "hemostats":               2.8,

    # pathological - infection risk
    "human_organs":            2.5,
    "blood_soaked_bandages":   2.5,

    # chemical - environmental hazard
    "disinfectant_bottles":    2.0,
    "ampoules_full":           2.0,
    "ampuoles_broken":         2.2,

    # pharmaceutical - ingestion risk
    "expired_tablets":         1.5,
    "vaccine_or_medicine_vials": 1.5,
    "syrup_bottles":           1.5,
    "iv_bottles":              1.5,

    # general waste - lower risk
    "used_masks":              1.2,
    "used_medical_gloves":     1.2,
    "uncategorized_or_overlapping": 1.3,
    "used_medical_paper":      1.0,
    "general_organic_waste":   1.0,
    "waterbottles":            1.0,
}

# CBAM is inserted after these stages of MobileNetV3-Large
CBAM_STAGES    = [3, 6, 12]
CBAM_REDUCTION = 16

# flags to control which variant to train
USE_CBAM = True
USE_HAFL = True
