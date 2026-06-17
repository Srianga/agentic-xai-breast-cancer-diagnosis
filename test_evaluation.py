from agents.evaluator import AgentEvaluator

evaluator = AgentEvaluator()

# Example based on your system
results = [
    {"prediction": 1, "probability": 0.89, "top_features": ["radius_mean", "texture_mean", "perimeter_mean"]},
    {"prediction": 0, "probability": 0.2, "top_features": ["radius_mean", "area_mean", "smoothness_mean"]},
    {"prediction": 1, "probability": 0.85, "top_features": ["radius_mean", "texture_mean", "perimeter_mean"]}
]

y_true = [1, 0, 1]
y_pred = [r["prediction"] for r in results]
probs = [r["probability"] for r in results]
features = [r["top_features"] for r in results]

print("Decision Accuracy:", evaluator.decision_accuracy(y_true, y_pred))
print("Explanation Consistency:", evaluator.explanation_consistency(features))
print("Trust Score:", evaluator.trust_score(probs))
print("Uncertainty Rate:", evaluator.uncertainty_rate(probs))