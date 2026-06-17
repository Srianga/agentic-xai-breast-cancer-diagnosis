import shap

class ExplanationAgent:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def run(self, x):
        explainer = shap.Explainer(self.model.predict, x)
        shap_values = explainer(x)

        shap_list = list(zip(self.features, shap_values.values[0]))
        shap_sorted = sorted(shap_list, key=lambda x: abs(x[1]), reverse=True)

        return [f[0] for f in shap_sorted[:3]]