
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone


from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold


class Reusable_Classifier:
    def __init__(self, model, feature_names=None):
        """Model wrapper class to handle training, evaluation,
        K Fold cross-validation, and saving/loading for any sklearn classifier

        Args:
            model (sklearn estimator): the model to be used
            feature_names (list of str): the names of the features in the model
        """
        self.model = model
        self.feature_names = feature_names

    # -----------------------------
    # Train function, takes in training data and fits the model
    # -----------------------------
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # -----------------------------
    # Predict probabilities
    # -----------------------------
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    # -----------------------------
    # Find best threshold (F1)
    # -----------------------------
    def find_best_threshold(self, y_true, y_proba):
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1)
        return thresholds[best_idx], precision[best_idx], recall[best_idx], f1[best_idx]

    # -----------------------------
    # Evaluate on test set, use one train test split,
    # and print classification report, confusion matrix, AUC, and feature importance (if available)
    # -----------------------------
    def evaluate(self, X_test, y_test):
        y_proba = self.predict_proba(X_test)

        best_thresh, prec, rec, f1 = self.find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= best_thresh).astype(int)

        print(f"Best threshold: {best_thresh}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1: {f1}\n")

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        auc = roc_auc_score(y_test, y_proba)
        print("ROC AUC:", auc)

        # Feature importance (RandomForest only)
        if hasattr(self.model, "feature_importances_") and self.feature_names is not None:
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            print(importance)

    # -----------------------------
    # Cross-validation, with best threshold selection on each fold, we use 5 folds,
    # each measured by AUC, precision, recall, and F1 score (with best threshold),
    # and we print out the mean ± std for each metric across folds at the end
    # -----------------------------
    def cross_validate(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        aucs, precisions, recalls, f1s = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = clone(self.model)
            model.fit(X_train, y_train)

            y_val_proba = model.predict_proba(X_val)[:, 1]

            thresh, prec, rec, f1 = self.find_best_threshold(y_val, y_val_proba)
            y_val_pred = (y_val_proba >= thresh).astype(int)

            auc = roc_auc_score(y_val, y_val_proba)

            aucs.append(auc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            print(f"Best threshold: {thresh}")
            print(f"Precision: {prec}")
            print(f"Recall: {rec}")
            print(f"F1: {f1}")
            print(
                f"Fold {fold} | AUC: {auc:.3f} | "
                f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}"
            )

        print("\n===== Cross-Validation Summary =====")
        print(f"AUC:       {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        print(f"Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
        print(f"F1 Score:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # -----------------------------
    # Save / load model using joblib
    # -----------------------------
    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

if __name__ == "__main__":
    ru = Reusable_Classifier(None)