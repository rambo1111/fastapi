from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model, label encoder, and symptoms list
clf = joblib.load("model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
all_symptoms = joblib.load("all_symptoms.joblib")

class SymptomsRequest(BaseModel):
    symptoms: List[str]

@app.post("/predict")
def get_prediction(request: SymptomsRequest):
    try:
        # Preprocess input
        input_df = pd.DataFrame(0, index=[0], columns=all_symptoms)
        symptoms_input = [symptom.strip() for symptom in request.symptoms]
        input_df.loc[0, symptoms_input] = 1

        # Predict disease
        disease_label = clf.predict(input_df)[0]
        predicted_disease = label_encoder.inverse_transform([disease_label])[0]

        return {"predicted_disease": predicted_disease}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
