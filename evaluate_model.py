from agents.agent import BreastCancerAgent

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import numpy as np


agent = BreastCancerAgent()


agent.llm.run = lambda *args, **kwargs: "LLM skipped"


# Load Dataset

data = load_breast_cancer()

X = data.data
y = data.target


# Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Predictions


y_pred = []
y_prob = []

for i in range(len(X_test)):

    sample = {
        f: float(X_test[i][idx])
        for idx, f in enumerate(agent.features)
    }

    result = agent.run(sample)

    pred = result["prediction"]


    mapped_pred = 1 - pred
    prob = result["probability"]

    mapped_prob = 1 - prob

    y_pred.append(mapped_pred)
    y_prob.append(mapped_prob)


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

# Decision Accuracy

decision_accuracy = accuracy

# Trust Score

confidence = np.mean([
    max(p, 1-p) for p in y_prob
])

human_agreement = decision_accuracy

trust_score = (confidence + human_agreement) / 2


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print Results

print("\n===== Evaluation Metrics =====\n")

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC Score      : {roc_auc:.4f}")
print(f"Decision Accuracy  : {decision_accuracy:.4f}")
print(f"Trust Score        : {trust_score:.4f}")

print("\n===== Confusion Matrix =====\n")

print(cm)