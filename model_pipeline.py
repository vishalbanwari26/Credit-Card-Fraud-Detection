import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import joblib

#Load dataset
df = pd.read_csv('creditcard.csv')
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df['Time'] = StandardScaler().fit_transform(df[['Time']])

#Split features and labels
X = df.drop('Class',axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y , stratify=y, test_size=0.2, random_state=42)

#Train isolation forest on normal data
X_train_non_fraud = X_train[y_train==0]
iso_forest = IsolationForest(contamination=0.001, random_state=42)
iso_forest.fit(X_train_non_fraud)
anomaly_scores = -iso_forest.decision_function(X_test)

#Train LightGBM
#scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
lgbm = LGBMClassifier(scale_pos_weight=10, random_state=42)
lgbm.fit(X_train, y_train)
lgbm_probs = lgbm.predict_proba(X_test)[:,1]

joblib.dump(lgbm, "lgbm_model.joblib")

#Combine scores
combined_score = 0.3*(anomaly_scores-anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()) + 0.7*lgbm_probs

#Save results
results_df = X_test.copy()
results_df['Fraud_TrueLabel']=y_test.values
results_df['LightGBM_Prob']=lgbm_probs
results_df['Anomaly_Score']=anomaly_scores
results_df['Combined_Score']=combined_score
results_df.to_csv('hybrid_results.csv', index=False)