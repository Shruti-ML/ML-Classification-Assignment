import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# Page setup
st.set_page_config(
    page_title="ML Model Comparison",
    page_icon="📊",
    layout="wide"
)


st.title("📊 ML Classification Model Comparison")

st.write(
    """
    This application evaluates five machine learning
    classification models using the uploaded test dataset.
    """
)


# Model paths
model_paths = {

    "Logistic Regression":
        "model/logistic_regression.joblib",

    "Decision Tree":
        "model/decision_tree.joblib",

    "kNN":
        "model/knn.joblib",

    "Naive Bayes":
        "model/naive_bayes.joblib",

    "Random Forest":
        "model/random_forest.joblib"
}


# Sidebar
st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a model",
    list(model_paths.keys())
)


# Upload
uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(
        uploaded_file
    )

    st.subheader("Uploaded Dataset")

    st.write(
        f"Rows: {data.shape[0]}"
    )

    st.write(
        f"Columns: {data.shape[1]}"
    )

    st.dataframe(
        data.head()
    )


    # Check target
    if "Diagnosis" not in data.columns:

        st.error(
            "Diagnosis column is missing."
        )

        st.stop()


    # Separate features and target
    X_test_app = data.drop(
        columns=["Diagnosis"]
    )

    y_test_app = data["Diagnosis"]


    # Load model
    model = joblib.load(
        model_paths[selected_model]
    )


    # Predict
    y_pred = model.predict(
        X_test_app
    )

    if hasattr(
        model,
        "predict_proba"
    ):

        y_score = model.predict_proba(
            X_test_app
        )[:, 1]

    else:

        y_score = model.decision_function(
            X_test_app
        )


    # Metrics
    accuracy = accuracy_score(
        y_test_app,
        y_pred
    )

    auc = roc_auc_score(
        y_test_app,
        y_score
    )

    precision = precision_score(
        y_test_app,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test_app,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test_app,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test_app,
        y_pred
    )


    # Display metrics
    st.subheader(
        f"Performance of {selected_model}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )


    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.4f}"
    )


    # Confusion matrix
    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test_app,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(fig)


    # Classification report
    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test_app,
        y_pred,
        target_names=[
            "Benign",
            "Malignant"
        ],
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4)
    )

else:

    st.info(
        "Upload test_data.csv to evaluate the selected model."
    )