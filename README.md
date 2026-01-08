# ECG Disease Detection from 12-Lead ECG (PTB-XL, PhysioNet)
<img width="2104" height="1228" alt="image" src="https://github.com/user-attachments/assets/8e11f5e2-bb0b-468e-a35f-639152d4bd45" />
<img width="2103" height="1355" alt="image" src="https://github.com/user-attachments/assets/52746e46-30d3-4316-a377-62fb88b77da9" />



End-to-end machine learning pipeline for automatic diagnosis of cardiac diseases and arrhythmias from 12-lead ECGs using the **PTB-XL** dataset (PhysioNet). The project trains deep neural networks on 10-second ECG recordings to predict multiple clinically relevant diagnoses and exposes them via a simple web app for ECG file upload and analysis.[1][2]

***

## Project Overview

This repository implements:

- Data ingestion and preprocessing for the **PTB-XL** ECG dataset (21,799 clinical 12-lead ECGs, 10 seconds each).[1]
- Label extraction from SCP-ECG statements (diagnostic subclasses) to build a **multi-label disease classifier** (22 common diagnoses, including MI variants, AFIB, LBBB, RBBB, conduction blocks, hypertrophy and rhythm disorders).[2]
- Training and evaluation of a 1D CNN model for **automatic ECG interpretation**, with ROC/AUC, F1, confusion matrices, and training curves stored for reporting.  
- Inference pipeline that accepts **arbitrary-length ECGs**, segments them into 10-second windows, aggregates predictions and produces a **clinical-style summary** (high/medium/low risk diagnoses).[1]
- A **Streamlit web app** that allows users to upload WFDB ECG records or select sample records from the dataset and view predicted diagnoses with probabilities.

The goal is to approximate an automated decision-support system similar to a modern ECG interpretation backend, focused on explainable, multi-diagnosis output.

***

## Dataset

This project uses the **PTB-XL ECG dataset** from **PhysioNet**.

- **Source:** PhysioNet: PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3)  
- **Records:** 21,799 clinical 12-lead ECGs from 18,869 patients  
- **Duration:** 10 seconds per recording  
- **Leads:** Standard 12-lead ECG (I, II, III, aVR, aVL, aVF, V1–V6)  
- **Sampling:** 500 Hz (full resolution, `records500/`) and 100 Hz downsampled (`records100/`)  
- **Annotations:** 71 SCP-ECG statements (diagnostic, form, rhythm), mapped into diagnostic classes and subclasses for ML use.[2]
- **Recommended splits:** 10 stratified folds (patient-wise), with folds 1–8 for training, 9 for validation, 10 for testing.

**Access & License**  
You must download the dataset yourself from PhysioNet and agree to their license:

- PTB-XL page: https://physionet.org/content/ptb-xl/1.0.3/  
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Required Citations (to include in reports/papers):**  

- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022).  
  *PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3).* PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kfzx-aw45  
- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020).  
  *PTB-XL: A large publicly available ECG dataset.* Scientific Data. https://doi.org/10.1038/s41597-020-0495-6  
- Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., et al. (2000).  
  *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23), e215–e220.

***

## Features

- **Multi-label ECG diagnosis:**
  - Predicts multiple diseases per ECG (e.g., AFIB + LBBB + IMI simultaneously).
  - Targets ~22 detailed diagnostic subclasses derived from SCP-ECG statements.[2]
- **Signal processing:**
  - Reads WFDB ECG records from PTB-XL (12 × 5000 samples at 500 Hz).
  - Pads/crops to fixed length for training; resamples arbitrary-length signals at inference.[1]
- **Model:**
  - 1D CNN on multi-lead time series (12-channel convolutional network).
  - Trained with BCEWithLogitsLoss for multi-label classification.
- **Evaluation:**
  - Macro AUC and macro F1 across all target diagnoses.
  - Per-diagnosis ROC curves and confusion matrices.
  - Per-class metrics (precision, recall, F1, AUC) exported as CSV for reporting.
- **Deployment:**
  - Inference accepts WFDB records of any length.
  - Segments into overlapping 10-second windows, aggregates predictions (max over time).
  - Streamlit app with:
    - File upload (.hea + .dat).
    - Sample selection from local PTB-XL `records500/`.
    - Ranked diagnosis list and probability bar chart.
    - Simple risk-stratified textual summary.

***

## Repository Structure

```text
ecg_project/
  README.md                 # This file
  requirements.txt          # Python dependencies
  .venv/                    # (optional) virtual environment

  data/
    ptbxl/
      ptbxl_database.csv    # PTB-XL metadata (one row per ECG)[file:1]
      scp_statements.csv    # SCP-ECG code metadata & mappings[file:2]
      records500/           # 500 Hz WFDB ECG files from PhysioNet
        00000/
          00001_hr.dat
          00001_hr.hea
          ...
        00001/
        ...
      records100/           # 100 Hz downsampled WFDB ECGs (optional)

  src/
    __init__.py
    config.py               # Paths, constants, target diagnosis list
    data_loading.py         # Metadata loading, SCP mapping, Dataset class
    preprocessing.py        # (Optional) signal utilities if separated
    models.py               # ECGConvNet 1D CNN model definition
    train.py                # Training loop, history logging, training plots
    evaluate.py             # Test metrics, ROC, confusion matrices, CSV export
    infer.py                # Arbitrary-length ECG segmentation & prediction
    utils.py                # (Optional) shared helpers

  saved_models/
    detailed_model.pt
    detailed_label_binarizer.pkl
    detailed_history.json
    detailed_training_curves.png
    detailed_roc_curves.png
    detailed_confusion_matrices.png
    detailed_performance_summary.csv
    detailed_classification_report.txt

  app/
    app.py

  notebooks/
    exploration.ipynb
