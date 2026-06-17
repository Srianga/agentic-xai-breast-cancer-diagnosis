from agents.agent import BreastCancerAgent
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

agent = BreastCancerAgent()

# 🔥 Disable LLM
agent.llm.run = lambda *args, **kwargs: "LLM skipped for evaluation"

data = load_breast_cancer()
X = data.data
y = data.target

# ✅ Proper split (IMPORTANT FIX)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

correct = 0
results = []

# ✅ Case Study (first 20 test samples)
for i in range(20):
    sample = {f: float(X_test[i][idx]) for idx, f in enumerate(agent.features)}
    result = agent.run(sample)

    pred = result["prediction"]
    actual = y_test[i]

    # 🔥 Fix label mapping
    mapped_pred = 1 - pred

    pred_label = "Malignant" if mapped_pred == 0 else "Benign"
    decision = "Biopsy" if mapped_pred == 0 else "Monitor"

    correct_flag = mapped_pred == actual

    if correct_flag:
        correct += 1

    results.append((i+1, pred_label, decision, correct_flag))


# ✅ Print Table (Case Study)
print("\n===== Simulation Results (Case Study) =====\n")

print(f"{'Case':<6} | {'Prediction':<12} | {'Agent Decision':<16} | {'Correct'}")
print("-" * 60)

for r in results:
    correct_symbol = "✔" if r[3] else "✘"
    print(f"{r[0]:<6} | {r[1]:<12} | {r[2]:<16} | {correct_symbol}")

print("-" * 60)

decision_accuracy = correct / 20
print(f"Decision Accuracy (Sample) = {correct}/20 = {decision_accuracy:.2f}")


# 🔥 FULL TEST EVALUATION (REAL RESULT)
full_correct = 0

for i in range(len(X_test)):
    sample = {f: float(X_test[i][idx]) for idx, f in enumerate(agent.features)}
    result = agent.run(sample)

    pred = result["prediction"]
    mapped_pred = 1 - pred

    if mapped_pred == y_test[i]:
        full_correct += 1

axai_accuracy = full_correct / len(X_test)

print("\n===== Full Model Evaluation =====")
print(f"AXAI Accuracy: {axai_accuracy:.4f}")


# 🔥 BASELINE MODEL (Logistic Regression)
baseline_model = LogisticRegression(max_iter=5000)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

print(f"Baseline (Logistic Regression) Accuracy: {baseline_acc:.4f}")