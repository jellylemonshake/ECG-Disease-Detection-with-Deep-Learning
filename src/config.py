import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "ptbxl"

PTBXL_DB_CSV = DATA_DIR / "ptbxl_database.csv"
PTBXL_SCP_CSV = DATA_DIR / "scp_statements.csv"

SAMPLING_FREQ = 500
ECG_LENGTH_SEC = 10
N_SAMPLES = SAMPLING_FREQ * ECG_LENGTH_SEC  # 5000
N_LEADS = 12

# DETAILED DIAGNOSTIC SUBCLASSES (doctor-level)
TARGET_DIAGNOSES = [
    'NORM',      # Normal ECG
    'LVH',       # Left ventricular hypertrophy
    'RBBB',      # Right bundle branch block
    'LBBB',      # Left bundle branch block
    'AFIB',      # Atrial fibrillation
    'SBRAD',     # Sinus bradycardia
    'IMI',       # Inferior myocardial infarction
    'AMI',       # Anterior myocardial infarction
    'LMI',       # Lateral myocardial infarction
    'STTC',      # ST/T changes
    'IRBBB',     # Incomplete RBBB
    'VESC',      # Ventricular escape rhythm
    'ST_Anterior', # Anterior ST elevation
    'T_INV',     # T-wave inversion
    'LAFB',      # Left anterior fascicular block
    'LPFB',      # Left posterior fascicular block
    'NSR',       # Normal sinus rhythm
    'SVTAC',     # Supraventricular tachycardia
    'WPW',       # Wolff-Parkinson-White
    'AVB',       # AV block
    'PAC',       # Premature atrial contraction
    'PVC'        # Premature ventricular contraction
]

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLDS = [9]
TEST_FOLDS = [10]

MODEL_SAVE_PATH = BASE_DIR / "saved_models" / "detailed_model.pt"
LABEL_BIN_PATH = BASE_DIR / "saved_models" / "detailed_label_binarizer.pkl"
CONFIG_JSON_PATH = BASE_DIR / "saved_models" / "detailed_config.json"
