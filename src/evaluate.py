import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve,
    auc,
    f1_score,
    precision_recall_fscore_support
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .config import TEST_FOLDS, MODEL_SAVE_PATH, LABEL_BIN_PATH, N_LEADS, TARGET_DIAGNOSES
from .data_loading import load_metadata, map_scp_to_detailed_diagnoses, make_splits, PTBXLDataset
from .models import ECGConvNet

def evaluate(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load test data
    print("Loading test data...")
    df_meta, scp_df = load_metadata()
    df_meta = map_scp_to_detailed_diagnoses(df_meta, scp_df)
    _, _, test_df = make_splits(df_meta, [], [], TEST_FOLDS)
    print(f"Test set size: {len(test_df)} records")

    mlb = load(LABEL_BIN_PATH)
    test_ds = PTBXLDataset(test_df, mlb)
    test_loader = DataLoader(
        test_ds, 
        batch_size=64, 
        shuffle=False, 
        num_workers=0,  # FIXED: Windows compatibility
        pin_memory=False
    )

    # 2. Load trained model
    model = ECGConvNet(n_leads=N_LEADS, n_classes=len(TARGET_DIAGNOSES)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 3. Generate predictions
    print("Running inference on test set...")
    all_targets = []
    all_probs = []
    
    test_pbar = tqdm(test_loader, desc="Test inference")
    with torch.no_grad():
        for x, y in test_pbar:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Binary predictions at 0.5 threshold
    preds_bin = (all_probs >= 0.5).astype("int")

    # 4. Calculate metrics
    auc_macro = roc_auc_score(all_targets, all_probs, average="macro")
    f1_macro = f1_score(all_targets, preds_bin, average="macro", zero_division=0)
    
    print(f"\n=== TEST SET PERFORMANCE (Detailed 22 Diagnoses) ===")
    print(f"Macro AUC:  {auc_macro:.4f}")
    print(f"Macro F1:   {f1_macro:.4f}")

    # Save directory
    save_dir = MODEL_SAVE_PATH.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5. Classification Report
    report = classification_report(all_targets, preds_bin, target_names=TARGET_DIAGNOSES, zero_division=0)
    print("\nClassification Report (Top Classes):\n")
    print(report)
    
    with open(save_dir / "detailed_classification_report.txt", "w") as f:
        f.write(report)

    # 6. ROC Curves
    plt.figure(figsize=(15, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, len(TARGET_DIAGNOSES)))
    
    for i, (class_name, color) in enumerate(zip(TARGET_DIAGNOSES, colors)):
        fpr, tpr, _ = roc_curve(all_targets[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Detailed ECG Diagnosis (22 Classes)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "detailed_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Detailed ROC curves saved: detailed_roc_curves.png")

    # 7. Confusion Matrices (Top 12 classes)
    n_plots = min(12, len(TARGET_DIAGNOSES))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(n_plots):
        class_name = TARGET_DIAGNOSES[i]
        cm = confusion_matrix(all_targets[:, i], preds_bin[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   cbar=False, square=True)
        axes[i].set_title(f'{class_name}\n(Support: {all_targets[:,i].sum()})', 
                         fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Confusion Matrices - Top Detailed Diagnoses', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "detailed_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Detailed confusion matrices saved: detailed_confusion_matrices.png")

    # 8. Performance Table
    perf_data = []
    for i, class_name in enumerate(TARGET_DIAGNOSES):
        support = all_targets[:, i].sum()
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets[:, i], preds_bin[:, i], average='binary', zero_division=0
        )
        roc_auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        perf_data.append({
            'Diagnosis': class_name,
            'Support': support,
            'Precision': f'{precision:.3f}',
            'Recall': f'{recall:.3f}',
            'F1': f'{f1:.3f}',
            'AUC': f'{roc_auc:.3f}'
        })
    
    perf_df = pd.DataFrame(perf_data)
    print("\nDetailed Performance Summary (Top 10):\n")
    print(perf_df.head(10).to_string(index=False))
    perf_df.to_csv(save_dir / "detailed_performance_summary.csv", index=False)
    print("✓ Detailed performance table saved: detailed_performance_summary.csv")

    print(f"\n=== All detailed visualizations saved to: {save_dir} ===")

if __name__ == "__main__":
    evaluate()
