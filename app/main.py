from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

try:
    model = joblib.load("logistic_final_titanic.pkl")
except FileNotFoundError:
    model = None
    print("❌ Model not found. Đảm bảo file logistic_final_titanic.pkl tồn tại.")

app = FastAPI(
    title="Titanic Survived Prediction API",
    description="API dự đoán hành khách sống sót trên tàu Titanic",
)

class PassengerInput(BaseModel):
    Age: float
    Fare: float
    Sex: str              
    Pclass: int           
    Embarked: str         
    Title: str            
    Family_Cat: str      

    class Config:
        schema_extra = {
            "example": {
                "Age": 29.0,
                "Fare": 50.0,
                "Sex": "female",
                "Pclass": 1,
                "Embarked": "S",
                "Title": "Mr",
                "Family_Cat": "SmallFamily"
            }
        }

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với Titanic API! Truy cập /docs để thử nghiệm."}

@app.post("/predict")
def predict(passenger: PassengerInput):
    if model is None:
        return {"error": "Model không được load."}

    input_df = pd.DataFrame([passenger.dict()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(probability, 3),
        "result": "✅ Sống sót" if prediction == 1 else "❌ Không sống sót"
    }
