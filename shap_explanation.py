import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

#Load data
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y=df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#Train model
model = LGBMClassifier()
model.fit(X_train, y_train)

#SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

#Visualize first sample
shap.plots.waterfall(shap_values[0])
