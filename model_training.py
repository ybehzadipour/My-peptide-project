"""
Project: Peptide Function Prediction (Ensemble Architecture)
Author: Yasaman Behzadipour, Pharm.D.
Description: 
    This script implements a Voting Ensemble (Stacking) approach to predict 
    peptide bioactivity (e.g., Cell-Penetrating Peptides). 
    It leverages 566 physicochemical features extracted via AAindex1.
    
    Models used:
    1. Random Forest (RF)
    2. Support Vector Machine (SVM)
    3. K-Nearest Neighbors (KNN)
    4. Decision Tree (DT)
    5. Naive Bayes (NB)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("🚀 Loading Dataset...")

# Load Sequences & Labels
# (Assuming 'DupDroped.csv' contains Sequence and Class columns)
try:
    df_seq = pd.read_csv('DupDroped.csv') # Reads header automatically
    y = df_seq.iloc[:, 1].values # Second column is the Class/Label
except:
    # Fallback if header is missing
    df_seq = pd.read_csv('DupDroped.csv', header=None)
    y = df_seq.iloc[:, 1].values

# Load Features (The AAindex calculations)
X = pd.read_csv('Output.csv')
if 'Unnamed: 0' in X.columns:
    X = X.drop(columns=['Unnamed: 0']) # Clean index column if present

# Alignment Check
if len(X) != len(y):
    print(f"⚠️ Warning: Data mismatch! X={len(X)}, y={len(y)}")
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y[:min_len]

# Handling Missing Values (Imputation) & Scaling
# Scaling is crucial for KNN and SVM to work correctly
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
le = LabelEncoder()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
y_encoded = le.fit_transform(y.astype(str)) # Converts 'ACP'/'Non-ACP' to 0/1

print(f"✅ Data Ready: {X_scaled.shape} samples processed.")

# ==========================================
# 2. MODEL DEFINITION (The "Five Pillars")
# ==========================================
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('SVM', SVC(probability=True, random_state=42)), # Probability=True needed for Soft Voting
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Naive Bayes', GaussianNB())
]

# The Ensemble (Voting Classifier)
# We use 'soft' voting to average the probabilities of all models
ensemble = VotingClassifier(estimators=models, voting='soft')

# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================
# Stratified Split ensures we keep the same % of positive samples in Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n📊 Benchmarking Individual Models:")
for name, model in models:
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"   • {name:<15} Accuracy: {acc:.2%}")

print("\n🏆 Training Ensemble Model...")
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
final_acc = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"✅ ENSEMBLE ACCURACY: {final_acc:.2%}")
print("-" * 40)

# ==========================================
# 4. VISUALIZATION
# ==========================================
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Ensemble Confusion Matrix')
plt.show()
