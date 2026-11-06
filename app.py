import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.caption("Machine learning project • Streamlit dashboard")
st.info(
    "This is an educational demonstration. The included dataset is synthetic, not real banking data. "
    "Predictions must not be used to make real financial decisions."
)

@st.cache_data
def load_demo_data():
    return pd.read_csv("data/sample_creditcard.csv")

def prepare_data(data):
    if "Class" not in data.columns:
        raise ValueError("The dataset must contain a target column named 'Class' (0 = genuine, 1 = fraud).")
    if data["Class"].isna().any():
        raise ValueError("The Class column contains missing values.")
    y = pd.to_numeric(data["Class"], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")
    X = data.drop(columns=["Class"]).copy()
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric feature columns were found.")
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().all().any():
        X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No usable numeric feature columns remain.")
    return X, y.astype(int)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    st.caption("Required target column: Class (0 = genuine, 1 = fraud).")
    use_demo = st.checkbox("Use included demo dataset", value=uploaded is None)

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        data_label = uploaded.name
    elif use_demo:
        raw = load_demo_data()
        data_label = "Included synthetic demo dataset"
    else:
        st.warning("Upload a CSV file or select the included demo dataset.")
        st.stop()

    X, y = prepare_data(raw)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        st.error("The dataset needs at least two examples from each class.")
        st.stop()

    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(raw):,}")
    c2.metric("Features", X.shape[1])
    c3.metric("Genuine", f"{int((y == 0).sum()):,}")
    c4.metric("Fraud", f"{int((y == 1).sum()):,}")
    st.caption(f"Loaded: {data_label}")
    with st.expander("Preview dataset"):
        st.dataframe(raw.head(10), use_container_width=True)

    st.subheader("Class distribution")
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(["Genuine (0)", "Fraud (1)"], counts.values)
    ax.set_ylabel("Number of transactions")
    ax.set_title("Transaction classes")
    st.pyplot(fig)
    plt.close(fig)

    # Stratified split preserves the minority-class proportion in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ))
    ])
    with st.spinner("Training the model..."):
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    st.subheader("Model evaluation")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
    m4.metric("F1-score", f"{f1_score(y_test, pred, zero_division=0):.3f}")
    try:
        auc = roc_auc_score(y_test, probabilities)
        m5.metric("ROC-AUC", f"{auc:.3f}")
    except ValueError:
        m5.metric("ROC-AUC", "N/A")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks([0, 1], labels=["Genuine", "Fraud"])
        ax.set_yticks([0, 1], labels=["Genuine", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_test, pred, target_names=["Genuine", "Fraud"],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("Try a transaction")
    st.write("Enter values for the same feature columns used by the uploaded dataset.")
    st.caption("For the included demo data, the fields are Time, V1–V28 and Amount. Defaults are zero.")
    with st.form("transaction_form"):
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(X.columns):
            default = 0.0
            if feature == "Amount":
                default = 100.0
            values[feature] = cols[i % 3].number_input(
                feature, value=float(default), format="%.6f", key=f"feature_{feature}"
            )
        submitted = st.form_submit_button("Check transaction", type="primary")
    if submitted:
        row = pd.DataFrame([[values[col] for col in X.columns]], columns=X.columns)
        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(probability >= 0.5)
        if prediction == 1:
            st.error(f"Model result: Potential fraud • Estimated fraud probability: {probability:.1%}")
        else:
            st.success(f"Model result: Likely genuine • Estimated fraud probability: {probability:.1%}")
        st.caption("This is a model estimate, not a guarantee. A human review and stronger validation are needed in real systems.")

    st.download_button(
        "Download test predictions",
        pd.DataFrame({
            "actual_class": y_test.to_numpy(),
            "predicted_class": pred,
            "fraud_probability": probabilities
        }).to_csv(index=False).encode("utf-8"),
        file_name="fraud_test_predictions.csv",
        mime="text/csv"
    )

except Exception as exc:
    st.error(f"Could not process the dataset: {exc}")
    st.markdown("Check that your CSV has numeric feature columns and a **Class** column containing 0 and 1.")
