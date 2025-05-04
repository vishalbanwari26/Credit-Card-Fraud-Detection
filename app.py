import streamlit as st
import pandas as pd
import shap
import joblib
from lightgbm import LGBMClassifier

# Load model and data
st.title("💳 Hybrid Fraud Detection Dashboard")

# Load results and model
results_df = pd.read_csv("hybrid_results.csv")
model = joblib.load("lgbm_model.joblib")  # Save your LightGBM model separately using joblib
X_test = results_df.drop(columns=[
    "Fraud_TrueLabel", "LightGBM_Prob", "Anomaly_Score", "Combined_Score"
])

# SHAP explainer setup
explainer = shap.Explainer(model)

# Top suspicious transactions
st.write("##Top 50 Suspicious Transactions")
top = results_df.sort_values("Combined_Score", ascending=False).head(50)
st.dataframe(top)

# Select and explain one
selected_index = st.selectbox("Select transaction to explain", top.index)
selected_row = X_test.loc[selected_index:selected_index]

st.write("###Selected Transaction Features")
st.dataframe(selected_row)

# SHAP explanation
shap_vals = explainer(selected_row)
shap_dict = dict(zip(X_test.columns, shap_vals.values[0]))

def explain_shap_decision(shap_values):
    pos = []
    neg = []
    for feat, val in shap_values.items():
        if val > 1000:
            pos.append(f"**{feat}** had a strong positive impact (+{int(val)})")
        elif val > 100:
            pos.append(f"**{feat}** slightly supported the fraud prediction (+{int(val)})")
        elif val < -1000:
            neg.append(f"**{feat}** strongly reduced the fraud score ({int(val)})")
        elif val < -100:
            neg.append(f"**{feat}** had a minor reducing effect ({int(val)})")

    result = "#### 🤖 Explanation:\n\n"
    if pos:
        result += "This transaction was flagged because:**\n- " + "\n- ".join(pos) + "\n"
    if neg:
        result += "\nCounter-evidence came from:**\n- " + "\n- ".join(neg)
    return result

st.markdown(explain_shap_decision(shap_dict))
