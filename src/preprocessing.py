import wfdb
import numpy as np
from pathlib import Path
from .config import DATA_DIR, N_LEADS, N_SAMPLES

def load_ecg_record(filename_hr):
    """
    filename_hr: e.g. 'records500/00000/00001_hr' (as in ptbxl_database.csv)
    Returns: numpy array of shape (12, N_SAMPLES)
    """
    record_path = DATA_DIR / filename_hr  # no extension
    # wfdb.rdsamp returns (signals, fields)
    signals, fields = wfdb.rdsamp(str(record_path))

    # signals: (time, channels)
    signal = signals.astype("float32").T  # -> (channels, time)

    # sanity check: 12 leads
    if signal.shape[0] != N_LEADS:
        raise ValueError(f"Expected {N_LEADS} leads, got {signal.shape[0]} for {record_path}")

    # pad/crop to fixed length
    if signal.shape[1] < N_SAMPLES:
        pad_width = N_SAMPLES - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
    elif signal.shape[1] > N_SAMPLES:
        signal = signal[:, :N_SAMPLES]

    return signal
