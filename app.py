from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
# custom transformers live in model/transformers.py and will be imported
# on-demand only if unpickling fails and needs them registered on __main__.

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_svm_pipeline.pkl"
TEST_PATH = BASE_DIR / "data" / "test.csv"

# Note: we avoid global tricks at import time. If joblib.load raises an
# AttributeError during unpickling (missing classes created from notebooks),
# load_model will import the transformers and register them temporarily on
# __main__ and retry. This keeps the top-level code clean and predictable.

st.set_page_config(page_title="Activity Recognition", layout="wide")
st.title("Activity Recognition Model")
st.write("Predict activities from sensor data using the trained SVM pipeline.")


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except AttributeError:
        # Register legacy custom transformers on __main__ and retry.
        import sys

        from model import transformers as _t

        main = sys.modules.setdefault("__main__", sys.modules[__name__])
        main.DuplicateFeatureFilter = _t.DuplicateFeatureFilter
        main.VarianceFilter = _t.VarianceFilter
        main.CorrelationFilter = _t.CorrelationFilter
        main.GroupwisePCA = _t.GroupwisePCA

        return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


try:
    model = load_model()
    test_df = load_test_data()
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

# Initialize session state for sample index and last prediction
if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0  # 0 means 'no selection'
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_index" not in st.session_state:
    st.session_state.last_index = None

# ── Sidebar ──────────────────────────────────────────────
# Sidebar: explain auto-predict behavior; 0 = no selection
st.sidebar.header("Select Test Sample (0 = no selection, auto-predict on change)")

# slider controls st.session_state.sample_index and triggers prediction on change
def _on_sample_change():
    idx = st.session_state.sample_index
    # slider value 0 means no selection
    if int(idx) == 0:
        st.session_state.last_prediction = None
        st.session_state.last_index = None
        return
    # map slider value (1..N) to positional index (0..N-1)
    pos = int(idx) - 1
    predict_for_index(pos)
def predict_for_index(idx: int):
    # compute prediction for a positional index and store in session state
    idx = int(idx)
    row = test_df.iloc[idx]
    feats = row.drop(["Activity", "subject"], errors="ignore").to_frame().T
    try:
        pred = model.predict(feats)[0]
    except Exception:
        pred = None
    st.session_state.last_prediction = pred
    st.session_state.last_index = idx

if st.sidebar.button("Random sample & predict"):
    # pick a random row and convert its label index to positional index
    raw_idx = test_df.sample(1).index[0]
    pos_idx = test_df.index.get_loc(raw_idx)
    # set slider to pos+1 (since 0 = no selection)
    st.session_state.sample_index = int(pos_idx) + 1
    predict_for_index(int(pos_idx))
    st.rerun()

sample_index = st.sidebar.slider(
    "Sample index",
    0,
    len(test_df),
    st.session_state.sample_index,
)

if int(sample_index) != int(st.session_state.sample_index):
    st.session_state.sample_index = int(sample_index)
    if int(sample_index) == 0:
        st.session_state.last_prediction = None
        st.session_state.last_index = None
    else:
        predict_for_index(int(sample_index) - 1)

st.sidebar.write(f"Total samples: {len(test_df)}")

# ── Single Prediction ─────────────────────────────────────
# Interpret slider: 0 -> no selection; otherwise positional = slider-1
if int(sample_index) == 0:
    sample_row = None
else:
    pos = int(sample_index) - 1
    sample_row = test_df.iloc[pos]

st.subheader("Single Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Sample", "None" if sample_row is None else pos)
col2.metric("Subject", "-" if sample_row is None else int(sample_row.get("subject", 0)))
col3.metric("Actual", "Unknown" if sample_row is None else str(sample_row.get("Activity", "Unknown")))

# Display last prediction (updates when sample changes or random sample pressed)
if st.session_state.get("last_prediction") is not None:
    st.success(f"Prediction: **{st.session_state.last_prediction}**")
else:
    st.info("Prediction: No prediction yet — select a sample or press 'Random sample & predict'.")

if sample_row is not None:
    st.dataframe(sample_row.to_frame().T, width="stretch")
else:
    st.info("No sample selected.")
st.divider()

# ── Test Set Evaluation ───────────────────────────────────
st.subheader("Test Set Evaluation")

y_true = test_df["Activity"]
X_test = test_df.drop(columns=["Activity", "subject"], errors="ignore")
y_pred = model.predict(X_test)
labels = sorted(y_true.unique())

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy",  f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall",    f"{rec:.3f}")
col4.metric("F1-Score",  f"{f1:.3f}")

# Confusion Matrix
st.write("**Confusion Matrix**")
cm = confusion_matrix(y_true, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.colorbar(im, ax=ax)

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
st.pyplot(fig, width="stretch")

# Classification Report
st.write("**Classification Report**")
report_df = pd.DataFrame(
    classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
).T
st.dataframe(report_df.round(3), width="stretch")