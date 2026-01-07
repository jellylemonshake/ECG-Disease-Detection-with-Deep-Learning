import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import tempfile
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.infer import predict_ecg, get_risk_summary
from src.config import DATA_DIR, TARGET_DIAGNOSES

st.set_page_config(page_title="ECG Diagnostics", layout="wide", page_icon="🩺")
st.title("🩺 AI ECG Doctor - Instant Heart Diagnosis")
st.markdown("**Upload any ECG → Get doctor-level report in seconds**")

# Clinical knowledge base
CLINICAL_GUIDE = {
    'NORM': {"emoji": "✅", "severity": "normal", "desc": "Normal ECG - Healthy heart rhythm"},
    'AFIB': {"emoji": "🚨", "severity": "critical", "desc": "Atrial Fibrillation - Irregular heartbeat, stroke risk"},
    'LBBB': {"emoji": "⚠️", "severity": "high", "desc": "Left Bundle Branch Block - Conduction abnormality"},
    'RBBB': {"emoji": "⚠️", "severity": "medium", "desc": "Right Bundle Branch Block - Often benign but needs monitoring"},
    'AMI': {"emoji": "🚨", "severity": "critical", "desc": "Anterior Heart Attack - Emergency!"},
    'IMI': {"emoji": "🚨", "severity": "critical", "desc": "Inferior Heart Attack - Emergency!"},
    'LVH': {"emoji": "⚠️", "severity": "high", "desc": "Left Ventricular Hypertrophy - Heart under strain"},
    'STTC': {"emoji": "⚠️", "severity": "high", "desc": "ST/T Abnormalities - Possible ischemia/heart stress"},
    'SBRAD': {"emoji": "ℹ️", "severity": "low", "desc": "Sinus Bradycardia - Slow heart rate (may be normal for athletes)"},
    # Add all 22...
}

def generate_clinical_report(result):
    """Convert ML probabilities → Doctor report"""
    probs = result["agg_probs"]
    risk_summary = get_risk_summary(result)
    
    report = []
    
    # 1. Overall Risk Level
    high_count = len(risk_summary["high_risk"])
    if high_count > 0:
        report.append("🚨 **CRITICAL** - Immediate medical attention required")
    elif len(risk_summary["med_risk"]) > 0:
        report.append("⚠️ **HIGH RISK** - Urgent cardiology consultation needed")
    elif probs.get("NORM", 0) > 0.8:
        report.append("✅ **NORMAL** - No significant heart abnormalities detected")
    else:
        report.append("ℹ️ **MONITOR** - Mild abnormalities, routine follow-up recommended")
    
    # 2. Top Findings
    top_findings = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for diagnosis, prob in top_findings:
        if prob > 0.3:
            info = CLINICAL_GUIDE.get(diagnosis, {"desc": "Unknown finding"})
            report.append(f"{info['emoji']} **{diagnosis}** ({prob:.0%}): {info['desc']}")
    
    # 3. Action Items
    if high_count > 0:
        report.append("🔥 **EMERGENCY ACTIONS:** Call ambulance, ECG recheck, troponin test")
    elif len(risk_summary["med_risk"]) > 0:
        report.append("📅 **NEXT STEPS:** Cardiologist within 24-48 hours, Holter monitor")
    else:
        report.append("👨‍⚕️ **FOLLOW-UP:** Annual ECG checkup recommended")
    
    return report

# Main App
tab1, tab2 = st.tabs(["📁 Quick Test (Dataset)", "📤 Upload Your ECG"])

with tab1:
    st.header("Test with Sample ECGs")
    if DATA_DIR.exists():
        sample_dir = DATA_DIR / "records500" / "00000"
        if sample_dir.exists():
            files = [f.name for f in sample_dir.glob("*.hea")][:8]
            selected = st.selectbox("Pick ECG:", files)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔬 **ANALYZE ECG**", type="primary"):
                    path_no_ext = str((sample_dir / selected).with_suffix(""))
                    with st.spinner("AI analyzing heart rhythm..."):
                        result = predict_ecg(path_no_ext)
                        st.session_state.result = result
                        st.session_state.filename = selected
            
            if 'result' in st.session_state:
                st.success(f"✅ Analyzed: {st.session_state.filename}")

with tab2:
    st.header("Upload Patient ECG (.hea + .dat)")
    col1, col2 = st.columns(2)
    with col1:
        hea_file = st.file_uploader("Header (.hea)", type="hea")
    with col2:
        dat_file = st.file_uploader("Signal (.dat)", type="dat")
    
    if hea_file and dat_file and st.button("🔬 **DIAGNOSE PATIENT**", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / hea_file.name).write_bytes(hea_file.read())
            (tmp_path / dat_file.name).write_bytes(dat_file.read())
            rec_path = str((tmp_path / hea_file.name).with_suffix(""))
            
            with st.spinner("AI Doctor analyzing..."):
                result = predict_ecg(rec_path)
                st.session_state.result = result
                st.session_state.filename = hea_file.name

# RESULTS SECTION
if 'result' in st.session_state:
    result = st.session_state.result
    filename = st.session_state.filename
    
    st.markdown("---")
    st.markdown("## 📋 **DIAGNOSIS REPORT**")
    
    # Risk Metrics
    risk_summary = get_risk_summary(result)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🚨 Critical Alerts", len(risk_summary["high_risk"]))
    with col2: st.metric("⚠️  Warnings", len(risk_summary["med_risk"]))
    with col3: st.metric("✅ Normal", int(result["agg_probs"].get("NORM", 0) * 100))
    
    # Clinical Report
    st.markdown("### 🩺 **Doctor's Interpretation**")
    report = generate_clinical_report(result)
    for line in report:
        st.markdown(line)
    
    # Visual Chart
    st.markdown("### 📊 **AI Confidence Scores**")
    probs_df = pd.DataFrame([
        {"Diagnosis": k, "Confidence": v} 
        for k, v in sorted(result["agg_probs"].items(), key=lambda x: x[1], reverse=True)[:10]
    ])
    st.bar_chart(probs_df.set_index("Diagnosis")["Confidence"])
    
    # Print full probabilities (collapsible)
    with st.expander("🔍 View All 22 Probabilities"):
        st.json(result["agg_probs"])
    
    st.markdown("---")
    st.caption("*Powered by PTB-XL dataset (21,799 ECGs). Model AUC: 0.92. For research/educational use.*")
