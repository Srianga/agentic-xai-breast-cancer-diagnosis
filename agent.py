import joblib
import numpy as np

from agents.diagnosis_agent import DiagnosisAgent
from agents.explanation_agent import ExplanationAgent
from agents.decision_agent import DecisionAgent
from agents.llm_agent import LLMReasoningAgent 


class BreastCancerAgent:   
    def __init__(self):
        self.model = joblib.load("model/xgb_model.pkl")
        self.scaler = joblib.load("model/scaler.pkl")
        self.features = joblib.load("model/features.pkl")

        self.diagnosis = DiagnosisAgent(self.model)
        self.explanation = ExplanationAgent(self.model, self.features)
        self.decision = DecisionAgent()
        self.llm = LLMReasoningAgent()   

    def safe_float(self, val):
        if isinstance(val, str):
            val = val.strip("[]")
        return float(val)

    def preprocess(self, data: dict):
        missing = [f for f in self.features if f not in data]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        x = np.array([[self.safe_float(data[f]) for f in self.features]])
        return self.scaler.transform(x)

    def run(self, data: dict):
        x = self.preprocess(data)

        pred, prob = self.diagnosis.run(x)
        top_features = self.explanation.run(x)
        decision = self.decision.run(pred, prob, top_features)

   
        llm_explanation = self.llm.run(pred, prob, top_features, decision)

        return {
            "prediction": pred,
            "probability": float(prob),
            "top_features": top_features,
            "decision": decision,
            "llm_explanation": llm_explanation   
        }