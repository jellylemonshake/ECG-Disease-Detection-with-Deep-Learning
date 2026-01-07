import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from joblib import dump
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS,
    MODEL_SAVE_PATH, LABEL_BIN_PATH, CONFIG_JSON_PATH,
    N_LEADS, TARGET_DIAGNOSES
)
from .data_loading import load_metadata, map_scp_to_detailed_diagnoses, make_splits, PTBXLDataset, create_label_binarizer
from .models import ECGConvNet

def train_model(
    batch_size=64,
    lr=1e-3,
    num_epochs=20,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and prepare data
    print("Loading metadata...")
    df_meta, scp_df = load_metadata()
    df_meta = map_scp_to_detailed_diagnoses(df_meta, scp_df)

    train_df, val_df, test_df = make_splits(df_meta, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    mlb = create_label_binarizer(train_df)
    dump(mlb, LABEL_BIN_PATH)
    print(f"Label classes: {mlb.classes_.tolist()}")

    # Create datasets and dataloaders (FIXED: num_workers=0 for Windows)
    train_ds = PTBXLDataset(train_df, mlb)
    val_ds = PTBXLDataset(val_df, mlb)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # FIXED: Windows compatibility
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # FIXED: Windows compatibility
        pin_memory=False
    )

    # Initialize model, loss, optimizer
    model = ECGConvNet(n_leads=N_LEADS, n_classes=len(TARGET_DIAGNOSES)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_f1": [],
        "epochs": num_epochs
    }

    best_val_auc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - train")
        for x, y in train_pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []
        val_pbar = tqdm(val_loader, desc="val")
        with torch.no_grad():
            for x, y in val_pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(y.cpu().numpy())
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= len(val_loader.dataset)
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        
        try:
            # Calculate per-class AUC, ignore classes with no positive samples
            auc_scores = []
            for i in range(all_targets.shape[1]):
                if len(np.unique(all_targets[:, i])) > 1:  # Both 0 and 1 present
                    auc_scores.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
            
            if auc_scores:
                auc_macro = np.mean(auc_scores)
            else:
                auc_macro = 0.0
        except:
            auc_macro = 0.0

        preds_bin = (all_probs >= 0.5).astype("int")
        f1_macro = f1_score(all_targets, preds_bin, average="macro", zero_division=0)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc_macro={auc_macro:.4f} val_f1_macro={f1_macro:.4f}")

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(auc_macro)
        history["val_f1"].append(f1_macro)

        # Save best model
        if auc_macro > best_val_auc:
            best_val_auc = auc_macro
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved! Val AUC: {best_val_auc:.4f}")

    # Save training history
    history_path = MODEL_SAVE_PATH.parent / "detailed_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Generate Training Curves Plot
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Training & Validation Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot AUC
    plt.subplot(1, 3, 2)
    plt.plot(history["val_auc"], label="Val AUC", color="green", linewidth=2, marker='o')
    plt.axhline(y=best_val_auc, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_val_auc:.3f}')
    plt.title("Validation AUC", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Macro AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot F1
    plt.subplot(1, 3, 3)
    plt.plot(history["val_f1"], label="Val F1", color="orange", linewidth=2, marker='s')
    plt.title("Validation F1 Score", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODEL_SAVE_PATH.parent / "detailed_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved to saved_models/detailed_training_curves.png")

    print(f"\n=== FINAL RESULTS (Detailed 22 Diagnoses) ===")
    print(f"Best val AUC: {best_val_auc:.4f}")
    print(f"Final model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
