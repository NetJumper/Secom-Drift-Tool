import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

# Resolve project paths relative to this file
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Required data artifacts
FEATURES_PATH = DATA_DIR / "secom_features_clean.csv"
SCORES_PATH = DATA_DIR / "secom_anomaly_scores.csv"

# Optional artifacts (computed if missing)
DRIFT_PATH = DATA_DIR / "secom_drift_scores.csv"
IMPACT_PATH = DATA_DIR / "secom_anomaly_drift_impact.csv"
LABELS_PATH = DATA_DIR / "secom_labels_raw.csv"

# Streamlit page configuration
st.set_page_config(page_title="SECOM Drift & Anomaly Dashboard", layout="wide")


def load_csv(path: Path, required: bool = True, index_col=None):
    """Load a CSV file if it exists, optionally enforcing presence."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    return pd.read_csv(path, index_col=index_col)


@st.cache_data
def load_data():
    """Load required and optional datasets for the dashboard."""
    X = load_csv(FEATURES_PATH, required=True)
    scores = load_csv(SCORES_PATH, required=True)

    drift_df = load_csv(DRIFT_PATH, required=False, index_col=0)
    impact_df = load_csv(IMPACT_PATH, required=False, index_col=0)
    y = load_csv(LABELS_PATH, required=False)

    # Reduce labels to a single column if present
    if y is not None:
        y = y.iloc[:, 0]

    return X, scores, drift_df, impact_df, y


def split_baseline_current(X: pd.DataFrame, baseline_frac: float):
    """Split data into baseline and current segments based on index order."""
    split_idx = int(len(X) * baseline_frac)
    return X.iloc[:split_idx], X.iloc[split_idx:]


def ensure_sample_id(scores: pd.DataFrame) -> pd.DataFrame:
    """Attach a sample index to preserve ordering for visualization."""
    scores = scores.copy()
    if "sample_id" not in scores.columns:
        scores["sample_id"] = np.arange(len(scores))
    return scores


def compute_drift_scores(X_baseline: pd.DataFrame, X_current: pd.DataFrame) -> pd.DataFrame:
    """Compute sensor drift as absolute mean shift between periods."""
    drift_score = (X_current.mean() - X_baseline.mean()).abs()
    return drift_score.sort_values(ascending=False).to_frame(name="drift_score")


def compute_impact_scores(X: pd.DataFrame, scores: pd.DataFrame, drift_df: pd.DataFrame) -> pd.DataFrame:
    """Measure how strongly drifted sensors differentiate anomalous samples."""
    drift_threshold = drift_df["drift_score"].quantile(0.95)
    drifted_sensors = drift_df[drift_df["drift_score"] >= drift_threshold].index.tolist()

    anomalous_idx = scores[scores["is_anomaly"] == 1].index
    normal_idx = scores[scores["is_anomaly"] == 0].index

    mean_anom = X.loc[anomalous_idx, drifted_sensors].mean()
    mean_norm = X.loc[normal_idx, drifted_sensors].mean()

    impact = (mean_anom - mean_norm).abs().sort_values(ascending=False)
    return impact.to_frame(name="impact_score")


# Page title
st.title("Inspection Drift & Anomaly Detection Decision-Support Tool (SECOM)")

# Load all datasets and stop execution if required files are missing
try:
    X, scores, drift_df, impact_df, y = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
baseline_frac = st.sidebar.slider("Baseline fraction", 0.5, 0.9, 0.7, 0.05)
top_n = st.sidebar.slider("Top N sensors", 5, 25, 10, 1)

# Ensure samples are indexed for plotting
scores = ensure_sample_id(scores)

# ---------------- Anomaly Overview ----------------
st.subheader("1) Anomaly Overview")

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(scores, x="anomaly_score", nbins=40, title="Anomaly Score Distribution")
    st.plotly_chart(fig_hist, width="stretch")

with col2:
    fig_line = px.line(
        scores.sort_values("sample_id"),
        x="sample_id",
        y="anomaly_score",
        title="Anomaly Score Over Sample Order",
    )
    st.plotly_chart(fig_line, width="stretch")

st.markdown("**Top risky samples (highest anomaly scores):**")
top_anoms = scores.sort_values("anomaly_score", ascending=False).head(15)
st.dataframe(top_anoms, width="stretch")

threshold = scores["anomaly_score"].quantile(0.97)

fig_scatter = px.scatter(
    scores,
    x="sample_id",
    y="anomaly_score",
    color=scores["anomaly_score"] >= threshold,
    title="Anomalous vs Normal Samples",
    labels={"color": "Is Anomalous"},
)
st.plotly_chart(fig_scatter, width="stretch")

# Optional label reference (not used in modeling)
if y is not None:
    st.caption("Labels are shown for reference only; the model is unsupervised.")
    label_counts = pd.Series(y).value_counts().rename_axis("label").reset_index(name="count")
    fig_labels = px.bar(label_counts, x="label", y="count", title="Label Distribution (Reference Only)")
    st.plotly_chart(fig_labels, width="stretch")

# ---------------- Drift Detection ----------------
st.subheader("2) Drift Detection (Baseline vs Current)")

X_baseline, X_current = split_baseline_current(X, baseline_frac)

# Compute drift scores if not provided
if drift_df is None or "drift_score" not in drift_df.columns:
    drift_df = compute_drift_scores(X_baseline, X_current)

top_drift = drift_df["drift_score"].sort_values(ascending=False).head(top_n).reset_index()
top_drift.columns = ["sensor", "drift_score"]

fig_drift_bar = px.bar(
    top_drift,
    x="sensor",
    y="drift_score",
    title=f"Top {top_n} Drifted Sensors (Mean Shift)",
)
st.plotly_chart(fig_drift_bar, width="stretch")

sensor_choice = st.selectbox(
    "Select a sensor to inspect drift distribution",
    top_drift["sensor"].tolist(),
    index=0,
)

df_dist = pd.DataFrame(
    {
        "value": pd.concat([X_baseline[sensor_choice], X_current[sensor_choice]], ignore_index=True),
        "period": (["Baseline"] * len(X_baseline)) + (["Current"] * len(X_current)),
    }
)

fig_dist = px.histogram(
    df_dist,
    x="value",
    color="period",
    nbins=40,
    barmode="overlay",
    title=f"Distribution Shift: {sensor_choice}",
)
st.plotly_chart(fig_dist, width="stretch")

# ---------------- Drift ↔ Anomaly Impact ----------------
st.subheader("3) Drift ↔ Anomaly Impact (What drifted sensors differentiate anomalies?)")

# Compute impact scores if not provided
if impact_df is None or "impact_score" not in impact_df.columns:
    impact_df = compute_impact_scores(X, scores, drift_df)

top_impact = impact_df["impact_score"].sort_values(ascending=False).head(top_n).reset_index()
top_impact.columns = ["sensor", "impact_score"]

fig_impact_bar = px.bar(
    top_impact,
    x="sensor",
    y="impact_score",
    title=f"Top {top_n} Drifted Sensors Contributing to Anomalies (Mean Difference)",
)
st.plotly_chart(fig_impact_bar, width="stretch")

impact_sensor = st.selectbox(
    "Select a sensor to compare Normal vs Anomalous",
    top_impact["sensor"].tolist(),
    index=0,
)

anomalous_idx = scores[scores["is_anomaly"] == 1].index
normal_idx = scores[scores["is_anomaly"] == 0].index

df_box = pd.DataFrame(
    {
        "value": pd.concat(
            [X.loc[normal_idx, impact_sensor], X.loc[anomalous_idx, impact_sensor]],
            ignore_index=True,
        ),
        "group": (["Normal"] * len(normal_idx)) + (["Anomalous"] * len(anomalous_idx)),
    }
)

fig_box = px.box(df_box, x="group", y="value", title=f"Normal vs Anomalous: {impact_sensor}")
st.plotly_chart(fig_box, width="stretch")

st.success("Dashboard loaded successfully. This supports anomaly triage, drift monitoring, and interpretability.")
