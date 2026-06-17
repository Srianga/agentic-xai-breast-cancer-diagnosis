from agents.agent import BreastCancerAgent
from agents.human_loop import HumanInLoop

agent = BreastCancerAgent()
human = HumanInLoop()

# ✅ Your full input data
data = {
  "radius_mean": 14.5,
  "texture_mean": 20.1,
  "perimeter_mean": 95.2,
  "area_mean": 650.0,
  "smoothness_mean": 0.1,
  "compactness_mean": 0.2,
  "concavity_mean": 0.15,
  "concave points_mean": 0.08,
  "symmetry_mean": 0.18,
  "fractal_dimension_mean": 0.06,
  "radius_se": 0.5,
  "texture_se": 1.2,
  "perimeter_se": 3.5,
  "area_se": 45.0,
  "smoothness_se": 0.007,
  "compactness_se": 0.03,
  "concavity_se": 0.04,
  "concave points_se": 0.01,
  "symmetry_se": 0.02,
  "fractal_dimension_se": 0.003,
  "radius_worst": 16.0,
  "texture_worst": 25.0,
  "perimeter_worst": 105.0,
  "area_worst": 800.0,
  "smoothness_worst": 0.14,
  "compactness_worst": 0.3,
  "concavity_worst": 0.4,
  "concave points_worst": 0.15,
  "symmetry_worst": 0.25,
  "fractal_dimension_worst": 0.08
}

# 🔥 Run AI
result = agent.run(data)

# 🔥 Human validation
final_result = human.review(result)

print("\n===== FINAL OUTPUT =====")
print("Prediction:", final_result["prediction"])
print("Probability:", final_result["probability"])
print("Top Features:", final_result["top_features"])
print("Decision:", final_result["decision"])
print("LLM Explanation:\n", final_result["llm_explanation"])
print("Final Decision:", final_result["final_decision"])
print("Doctor Note:", final_result["doctor_note"])