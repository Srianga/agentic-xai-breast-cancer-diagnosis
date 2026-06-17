class DiagnosisAgent:
    def __init__(self, model):
        self.model = model

    def run(self, x):
        prob = self.model.predict_proba(x)[0][1]
        pred = int(prob >= 0.5)
        return pred, prob