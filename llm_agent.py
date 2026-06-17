from dotenv import load_dotenv
import os
from google import genai
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


class LLMReasoningAgent:
    def run(self, prediction, probability, features, decision):
        prompt = f"""
Prediction: {prediction}
Probability: {probability}
Features: {', '.join(features)}
Decision: {decision}

Explain the reasoning in simple medical terms.
"""

        # 🔁 Retry logic (handles temporary API failure)
        for _ in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                if response and hasattr(response, "text"):
                    return response.text

            except Exception as e:
                print("⚠️ LLM retry due to:", e)
                time.sleep(2)

        # 🛑 Fallback (VERY IMPORTANT)
        return f"""
Fallback Explanation:
The model predicts {'HIGH' if prediction==1 else 'LOW'} cancer risk with probability {round(probability,2)}.
Key influencing features include {', '.join(features)}.
Recommended action: {decision}
"""