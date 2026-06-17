from fastapi import FastAPI, Body
from agents.agent import BreastCancerAgent   

app = FastAPI()

agent = BreastCancerAgent()   

@app.get("/")
def home():
    return {"message": "Agentic Breast Cancer API Running"}


@app.post("/predict")
def predict(data: dict = Body(...)):
    try:
        missing = [f for f in agent.features if f not in data]
        if missing:
            return {"error": f"Missing features: {missing}"}

        clean_data = {}
        for f in agent.features:
            val = data[f]

            if isinstance(val, list):
                val = val[0]

            if isinstance(val, str):
                val = val.strip().replace("[", "").replace("]", "")

            clean_data[f] = float(val)

        result = agent.run(clean_data)

        return result

    except Exception as e:
        return {"error": str(e)}