# SECOM Drift & Anomaly Detection Dashboard

## Overview
This project implements an unsupervised machine-learning decision-support tool for detecting anomalous production samples and identifying process drift in high-dimensional manufacturing sensor data. The system supports manufacturing and process engineers by providing interpretable anomaly scores, drift metrics, and interactive visualizations through a Streamlit dashboard.

The application uses the publicly available SECOM manufacturing dataset and applies Isolation Forest–based anomaly detection alongside sensor-level drift analysis.

---

## Dataset Source

This project uses the **UCI SECOM** dataset from Kaggle:

https://www.kaggle.com/datasets/paresh2047/uci-semcom

Raw files are not committed to this repository (see `data/raw/`). Processed datasets used by the app are stored in `data/processed/`.

---

## Project Structure

project-root/
├── app/
│   └── app.py
├── data/
│   └── processed/
│       ├── secom_features_clean.csv
│       ├── secom_anomaly_scores.csv
│       ├── secom_drift_scores.csv          
│       ├── secom_anomaly_drift_impact.csv  
│       └── secom_labels_raw.csv             
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── anomaly_detection.ipynb
│   └── drift_analysis.ipynb
├── requirements.txt
└── README.md

---

## Setup and Execution (Windows 10)

### 1. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
Ensure required CSV files are present in `data/processed/`, then run:
```bash
streamlit run app/app.py
```

The dashboard will open in your default web browser.

---

## Using the Dashboard
- Review **Anomaly Overview** to inspect anomaly score distributions and identify high-risk samples.
- Use **Drift Detection** to examine sensors exhibiting the largest distribution shifts between baseline and current periods.
- Explore **Drift ↔ Anomaly Impact** to identify sensors that contribute most strongly to anomalous samples.
- Adjust baseline fraction and number of top sensors using the sidebar controls.

---

## Notes
- The model operates in an **unsupervised** manner and does not require labeled training data.