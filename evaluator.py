import numpy as np

class AgentEvaluator:

    def decision_accuracy(self, y_true, y_pred):
        correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
        return correct / len(y_true)

    def explanation_consistency(self, features_list):
        base = set(features_list[0])
        scores = []

        for f in features_list[1:]:
            overlap = len(base.intersection(set(f)))
            scores.append(overlap / len(base))

        return np.mean(scores) if scores else 1.0

    def trust_score(self, probabilities):
        scores = []
        for p in probabilities:
            if p > 0.8:
                scores.append(1)
            else:
                scores.append(p / 0.8)
        return np.mean(scores)

    def uncertainty_rate(self, probabilities):
        uncertain = [p for p in probabilities if 0.4 <= p <= 0.6]
        return len(uncertain) / len(probabilities)