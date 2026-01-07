import ast
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import wfdb
import numpy as np
from .config import PTBXL_DB_CSV, PTBXL_SCP_CSV, DATA_DIR, TARGET_DIAGNOSES, N_LEADS, N_SAMPLES

def load_metadata():
    df = pd.read_csv(PTBXL_DB_CSV)
    scp_df = pd.read_csv(PTBXL_SCP_CSV)
    scp_df = scp_df.set_index("Unnamed: 0")
    return df, scp_df

def map_scp_to_detailed_diagnoses(df_meta, scp_df):
    """Map SCP codes to specific diagnostic subclasses"""
    print("Mapping SCP codes to detailed diagnoses...")
    
    # Manual mapping based on PTB-XL diagnostic subclasses
    mapping = {
        # Normal
        'NORM': ['NORM'],
        
        # Arrhythmias
        'AFIB': ['AFIB', 'AF'],
        'SBRAD': ['SBRAD', 'SB', 'NODD'],
        'NSR': ['NSR'],
        'SVTAC': ['SVTAC', 'AT'],
        'VESC': ['VESC'],
        'PAC': ['PAC', 'SVPC'],
        'PVC': ['PVC', 'VP'],
        'AVB': ['AVB', 'AB'],
        
        # Conduction defects
        'LBBB': ['LBBB'],
        'RBBB': ['RBBB'],
        'IRBBB': ['IRBBB'],
        'LAFB': ['LAFB'],
        'LPFB': ['LPFB'],
        'WPW': ['WPW'],
        
        # Myocardial Infarction
        'IMI': ['IMI'],
        'AMI': ['AMI'],
        'LMI': ['LMI'],
        
        # ST/T Changes
        'STTC': ['STTC'],
        'ST_Anterior': ['STE'],
        'T_INV': ['T_INV'],
        
        # Hypertrophy
        'LVH': ['LVH', 'LVHDDP']
    }
    
    # Reverse mapping: SCP code -> diagnosis
    scp_to_diag = {}
    for diag, scp_codes in mapping.items():
        for scp in scp_codes:
            if scp in scp_df.index:
                scp_to_diag[scp] = diag

    def extract_diagnoses(scp_codes_str):
        try:
            scp_codes = ast.literal_eval(scp_codes_str)
            diagnoses = set()
            for code, likelihood in scp_codes.items():
                if code in scp_to_diag and likelihood > 0:
                    diagnoses.add(scp_to_diag[code])
            return [d for d in diagnoses if d in TARGET_DIAGNOSES]
        except:
            return []

    df_meta["diagnoses"] = df_meta["scp_codes"].apply(extract_diagnoses)
    df_meta = df_meta[df_meta["diagnoses"].map(len) > 0].reset_index(drop=True)
    print(f"Records with detailed diagnoses: {len(df_meta)}")
    return df_meta

def load_ecg_record(filename_hr):
    """FIXED: Handle Windows tuple return from wfdb.rdsamp"""
    record_path = DATA_DIR / filename_hr
    record_name = str(record_path.with_suffix(''))
    
    try:
        # FIXED: Handle both object and tuple return formats
        record_data = wfdb.rdsamp(record_name)
        if isinstance(record_data, tuple):
            signal, _ = record_data  # Windows returns (signal, fields)
        else:
            signal = record_data.p_signals  # Linux/Mac returns object
        
        # Ensure correct shape: transpose to (channels, time)
        if signal.ndim == 2 and signal.shape[1] == N_LEADS:
            signal = signal.T  # (time, channels) -> (channels, time)
        elif signal.ndim == 2 and signal.shape[0] == N_LEADS:
            signal = signal  # Already correct
        
        signal = signal.astype("float32")
        
        # Pad/crop to exact length (12, 5000)
        if signal.shape[0] != N_LEADS:
            print(f"WARNING: Expected {N_LEADS} leads, got {signal.shape[0]}")
            # Pad with zeros if fewer leads
            if signal.shape[0] < N_LEADS:
                pad_channels = N_LEADS - signal.shape[0]
                signal = np.pad(signal, ((0, pad_channels), (0, 0)), mode="constant")
        
        if signal.shape[1] < N_SAMPLES:
            pad_width = N_SAMPLES - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
        elif signal.shape[1] > N_SAMPLES:
            signal = signal[:, :N_SAMPLES]
            
        return signal  # (12, 5000)
        
    except Exception as e:
        print(f"ERROR loading {filename_hr}: {e}")
        # Return zero-padded dummy signal
        return np.zeros((N_LEADS, N_SAMPLES), dtype="float32")

class PTBXLDataset(Dataset):
    def __init__(self, df_meta, mlb: MultiLabelBinarizer, transforms=None):
        self.df = df_meta.reset_index(drop=True)
        self.mlb = mlb
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = load_ecg_record(row["filename_hr"])
        if self.transforms:
            signal = self.transforms(signal)
        x = torch.from_numpy(signal).float()
        labels = row["diagnoses"]
        y = torch.from_numpy(self.mlb.transform([labels])[0].astype("float32"))
        return x, y

def make_splits(df_meta, train_folds, val_folds, test_folds):
    train_df = df_meta[df_meta["strat_fold"].isin(train_folds)].reset_index(drop=True)
    val_df = df_meta[df_meta["strat_fold"].isin(val_folds)].reset_index(drop=True)
    test_df = df_meta[df_meta["strat_fold"].isin(test_folds)].reset_index(drop=True)
    print(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def create_label_binarizer(df_meta):
    mlb = MultiLabelBinarizer(classes=TARGET_DIAGNOSES)
    mlb.fit(df_meta["diagnoses"])
    print(f"Label classes fitted: {mlb.classes_.tolist()}")
    return mlb
