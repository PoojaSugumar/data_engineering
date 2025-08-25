# 04_spending_classification.ipynb

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# Load cleaned finance data
df = pd.read_csv("../outputs/cleaned_finance_data.csv")

# Feature Engineering
df['Savings'] = df['Income'] - df['Expense']
df['Savings_Ratio'] = df['Savings'] / df['Income']
df['Weekend_Spending_Ratio'] = df['Weekend_Expense'] / df['Expense']
df['High_Value_Transaction'] = df['Expense'] > df['Expense'].quantile(0.90)
df['High_Value_Count'] = df.groupby('Month')['High_Value_Transaction'].transform('sum')
df['Category_Spending_Ratio'] = df['Category_Expense'] / df['Expense']

# Rule-based Labeling
def classify_behavior(row):
    if row['Savings_Ratio'] > 0.3 and row['High_Value_Count'] < 2:
        return 'Saver'
    elif row['Savings_Ratio'] < 0.1 and row['High_Value_Count'] > 5:
        return 'Impulsive'
    elif row['Savings_Ratio'] > 0.2:
        return 'Balanced'
    else:
        return 'Spender'

df['Spending_Behavior'] = df.apply(classify_behavior, axis=1)

# Prepare data for model training
features = ['Savings_Ratio', 'Weekend_Spending_Ratio', 'High_Value_Count', 'Category_Spending_Ratio']
X = df[features]
y = df['Spending_Behavior']

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Save model and predictions
os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)
joblib.dump(clf, "../models/classification_model.pkl")

# Save predictions
df['Predicted_Behavior'] = clf.predict(scaler.transform(X[features]))
df[['Month', 'Income', 'Expense', 'Savings_Ratio', 'High_Value_Count', 'Predicted_Behavior']].to_csv("../outputs/classified_behavior.csv", index=False)

print("Classification model and predictions saved successfully.")
