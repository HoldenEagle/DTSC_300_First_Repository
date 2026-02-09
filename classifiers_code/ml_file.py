from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from reusable_classifier import Reusable_Classifier

# -----------------------------
# Load in data from our premade csv file (which we made in the HAR reader file)
# -----------------------------
ml_df = pd.read_csv(r'C:\Users\holde\DTSC_First_Repo\DTSC_300_First_Repository\data\har_5min_features.csv')

FEATURES = [
    'hr_norm',
    'hr_norm_30m',
    'acc_mag_norm',
    'acc_mag_norm_30m',
    'acc_mag_norm_60m',
    'acc_mag_norm_480m',
    'hr_norm_60m',
    'hr_norm_480m'
]

X = ml_df[FEATURES]

y = ml_df['is_sleep_label']

# -----------------------------
# Train / test split (80/20) with stratification to maintain class balance in both sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Random Forest model with balanced class weights to handle any class imbalance, and some regularization parameters to prevent overfitting
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rc = Reusable_Classifier(model=rf, feature_names=FEATURES)

rc.train(X_train, y_train)

# Test-set evaluation (prints first big block)
rc.evaluate(X_test, y_test)

# Cross-validation (prints fold outputs + summary)
rc.cross_validate(X, y)


