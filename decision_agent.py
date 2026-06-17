class DecisionAgent:
    def run(self, pred, prob, top_features):
        if pred == 1 and prob > 0.8:
            return f"HIGH RISK → Immediate oncology consultation. Key factors: {top_features}"
        elif pred == 1:
            return f"MODERATE RISK → Further diagnostic tests recommended. Key factors: {top_features}"
        else:
            return f"LOW RISK → Routine monitoring. Key factors: {top_features}"