
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score, roc_auc_score

def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    recall_sco = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    classification_rep = classification_report(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(x_test)
    else:
        y_scores = None

    if y_scores is not None:
        roc_auc = roc_auc_score(y_test, y_scores)
    else:
        roc_auc = float("nan")

    print(f"=== ( {model_name} ) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall_sco:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if y_scores is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        print("ROC AUC: N/A (model does not support probability or decision function)")
    print("Classification Report:")
    print(classification_rep)
    print("#" * 100)

def plot_confusion_matrix(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(y_test)
    acc = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} \nAccuracy: {acc:.2%}")
    plt.show()




