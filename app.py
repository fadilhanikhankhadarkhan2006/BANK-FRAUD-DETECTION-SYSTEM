import streamlit as st
import pandas as pd
import joblib


# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# -------------------------------
# TITLE
# -------------------------------

st.title("💳 Bank Transaction Fraud Detection")

st.write(
    "Machine learning system for identifying potentially "
    "fraudulent financial transactions."
)


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.header("⚙️ Detection Settings")

threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.write(
    "Transactions with a fraud probability "
    "above this threshold will be flagged."
)


# -------------------------------
# FILE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Transactions")

    st.dataframe(
        data.head(100),
        use_container_width=True
    )


    # -------------------------------
    # REQUIRED COLUMNS
    # -------------------------------

    required_columns = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]


    if missing_columns:

        st.error(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    else:

        # -------------------------------
        # PREPARE DATA
        # -------------------------------

        X_new = data[required_columns].copy()


        # Scale Amount
        X_new["Amount"] = scaler.transform(
            X_new[["Amount"]]
        )


        # -------------------------------
        # PREDICTION
        # -------------------------------

        probabilities = model.predict_proba(
            X_new
        )[:, 1]


        # Apply selected threshold
        predictions = (
            probabilities >= threshold
        ).astype(int)


        # -------------------------------
        # ADD RESULTS
        # -------------------------------

        data["Fraud Probability"] = probabilities

        data["Prediction"] = predictions


        data["Result"] = data["Prediction"].map(
            {
                0: "Normal",
                1: "Potential Fraud"
            }
        )


        # -------------------------------
        # RISK LEVEL
        # -------------------------------

        def get_risk(probability):

            if probability >= 0.80:
                return "Critical"

            elif probability >= 0.50:
                return "High"

            elif probability >= 0.20:
                return "Medium"

            else:
                return "Low"


        data["Risk Level"] = [
            get_risk(probability)
            for probability in probabilities
        ]


        # -------------------------------
        # SUMMARY
        # -------------------------------

        total_transactions = len(data)

        fraud_count = (
            data["Prediction"] == 1
        ).sum()

        normal_count = (
            data["Prediction"] == 0
        ).sum()

        fraud_percentage = (
            fraud_count / total_transactions
        ) * 100


        st.subheader("📊 Detection Summary")


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )


        col2.metric(
            "Potential Fraud",
            f"{fraud_count:,}"
        )


        col3.metric(
            "Normal",
            f"{normal_count:,}"
        )


        col4.metric(
            "Flagged Rate",
            f"{fraud_percentage:.2f}%"
        )


        # -------------------------------
        # RISK DISTRIBUTION
        # -------------------------------

        st.subheader("⚠️ Risk Distribution")


        risk_counts = (
            data["Risk Level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Critical"],
                fill_value=0
            )
        )


        st.bar_chart(risk_counts)


        # -------------------------------
        # SUSPICIOUS TRANSACTIONS
        # -------------------------------

        st.subheader("🚨 Potentially Suspicious Transactions")


        suspicious = data[
            data["Prediction"] == 1
        ].sort_values(
            "Fraud Probability",
            ascending=False
        )


        if len(suspicious) > 0:

            st.dataframe(
                suspicious.head(100),
                use_container_width=True
            )

        else:

            st.success(
                "No transactions crossed the selected "
                "fraud probability threshold."
            )


        # -------------------------------
        # HIGH RISK TRANSACTIONS
        # -------------------------------

        st.subheader("🔴 High-Risk Transactions")


        high_risk = data[
            data["Risk Level"].isin(
                ["High", "Critical"]
            )
        ].sort_values(
            "Fraud Probability",
            ascending=False
        )


        if len(high_risk) > 0:

            st.dataframe(
                high_risk.head(100),
                use_container_width=True
            )

        else:

            st.info(
                "No High or Critical risk transactions found."
            )


        # -------------------------------
        # DOWNLOAD REPORT
        # -------------------------------

        st.subheader("📥 Download Report")


        csv = data.to_csv(
            index=False
        )


        st.download_button(
            label="Download Fraud Detection Report",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )


else:

    st.info(
        "Upload a transaction CSV file to start detection."
    )