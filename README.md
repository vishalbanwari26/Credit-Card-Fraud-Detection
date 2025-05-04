# 💳 Hybrid Fraud Detection System

This project implements a hybrid fraud detection pipeline using both supervised and unsupervised machine learning techniques on the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

It combines the precision of **LightGBM** with the anomaly-spotting capability of **Isolation Forest**, and wraps it all in an interactive **Streamlit dashboard** powered by **SHAP explanations** — to help humans understand *why* a transaction was flagged.

---

## 🔍 Key Features

- ⚙️ **Hybrid Model**: Combines LightGBM (supervised) and Isolation Forest (unsupervised) for robust fraud scoring
- 📊 **SHAP Explanations**: Visual + textual feature attribution for each flagged transaction
- 🌐 **Streamlit App**: Interactive dashboard for browsing, reviewing, and explaining fraud predictions

---

## 📁 Project Structure

```
├── model_pipeline.py          # Preprocessing, training, scoring, and CSV export
├── shap_explanation.py        # SHAP visualizations and analysis
├── app.py                     # Streamlit dashboard with explanation
├── genai_prompt.py            # Optional: Prompt template (not currently used)
├── lgbm_model.joblib          # Saved LightGBM model (generated after training)
├── hybrid_results.csv         # Scored output (generated after training)
```

---

## 📦 Installation

```bash
# Create environment (optional but recommended)
conda create -n fraud-detection python=3.10
conda activate fraud-detection

# Install dependencies
conda install pandas numpy scikit-learn lightgbm -y
pip install shap streamlit
```

---

## 📂 How to Run

### 1. Train the models and score the data
```bash
python model_pipeline.py
```

### 2. (Optional) Visualize SHAP waterfall plot for one prediction
```bash
python shap_explanation.py
```

### 3. Launch the interactive dashboard
```bash
streamlit run app.py
```

---

## 📊 SHAP Explanation

Each prediction is broken down by SHAP values and translated into plain English:

> **This transaction was flagged because:**  
> - V1 had a strong positive impact (+15991)  
> - V26 slightly supported the fraud prediction  
>  
> **Counter-evidence came from:**  
> - V4 strongly reduced the fraud score

This enables human analysts and auditors to understand *why* a transaction was flagged — critical for real-world fraud systems.

---

## 🤖 Why Not Deep Learning?

While deep learning is powerful, this project uses tree-based models (LightGBM) because:

- The dataset is **tabular**, where gradient-boosted trees usually outperform neural nets
- Tree models offer **faster training** and **better explainability** (via SHAP)
- Regulators and auditors require **transparent models**, which deep learning often lacks

---

## 📚 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions with anonymized PCA features
- Extremely imbalanced: ~0.17% fraud cases

---

## 🧩 Possible Extensions

- Add **autoencoders** or **LSTM models** for deep anomaly detection
- Integrate more advanced SHAP visualizations for teams
- Expand to **user-behavior modeling** or multi-transaction trails

---


## 📜 License

This project is released under the MIT License.
