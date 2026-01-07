import numpy as np
import torch
from joblib import load
import wfdb
from pathlib import Path

from .config import MODEL_SAVE_PATH, LABEL_BIN_PATH, N_LEADS, N_SAMPLES
from .models import ECGConvNet

def load_trained_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mlb = load(LABEL_BIN_PATH)
    model = ECGConvNet(n_leads=N_LEADS, n_classes=len(mlb.classes_)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model, mlb, device

def read_wfdb_any_length(path_no_ext: str):
    rec = wfdb.rdsamp(path_no_ext)
    sig = rec.p_signals.T.astype("float32")  # (ch, time)
    return sig, rec.fs

def segment_signal(sig, fs, target_fs=500, window_sec=10, overlap=0.5):
    """Segment arbitrary length ECG into overlapping 10s windows"""
    if fs != target_fs:
        from scipy.signal import resample
        sig = resample(sig, int(len(sig[0]) * target_fs / fs), axis=1)
    
    n_samples = sig.shape[1]
    win_len = int(window_sec * target_fs)
    step = int(win_len * (1 - overlap))
    
    if n_samples < win_len:
        pad = win_len - n_samples
        sig_pad = np.pad(sig, ((0, 0), (0, pad)), mode="constant")
        return [sig_pad]

    windows = []
    for start in range(0, n_samples - win_len + 1, step):
        end = start + win_len
        windows.append(sig[:, start:end])
    return windows

def predict_ecg(path_no_ext: str, threshold=0.5):
    """Main prediction function for arbitrary length ECG"""
    model, mlb, device = load_trained_model()
    sig, fs = read_wfdb_any_length(path_no_ext)
    windows = segment_signal(sig, fs)

    all_probs = []
    with torch.no_grad():
        for w in windows:
            # Ensure exact shape (12, 5000)
            if w.shape[1] != N_SAMPLES:
                if w.shape[1] < N_SAMPLES:
                    pad = N_SAMPLES - w.shape[1]
                    w = np.pad(w, ((0, 0), (0, pad)), mode="constant")
                else:
                    w = w[:, :N_SAMPLES]
            
            x = torch.from_numpy(w).unsqueeze(0).to(device).float()
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            all_probs.append(probs)

    all_probs = np.stack(all_probs, axis=0)
    
    # Aggregate: max probability across windows (detects if ANY segment is abnormal)
    agg_probs = all_probs.max(axis=0)
    
    # Predicted labels (above threshold)
    labels = mlb.classes_
    predicted = [labels[i] for i, p in enumerate(agg_probs) if p >= threshold]

    return {
        "agg_probs": dict(zip(labels, agg_probs.tolist())),
        "predicted_labels": predicted,
        "n_windows": len(windows),
        "window_length": N_SAMPLES,
        "sampling_rate": 500
    }

def get_risk_summary(result):
    """Clinical risk categorization"""
    probs = result["agg_probs"]
    high_risk = [k for k, v in probs.items() if v > 0.7 and k in ['AFIB', 'AMI', 'IMI', 'LMI', 'LBBB', 'RBBB', 'SVTAC']]
    med_risk = [k for k, v in probs.items() if 0.4 < v <= 0.7 and k not in high_risk]
    low_risk = [k for k, v in probs.items() if v > 0.3 and k not in high_risk + med_risk]
    
    return {
        "high_risk": high_risk,
        "med_risk": med_risk,
        "low_risk": low_risk
    }
