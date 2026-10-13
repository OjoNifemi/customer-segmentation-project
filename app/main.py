import joblib 
from fastapi import FastAPI  
from pydantic import BaseModel, Field  
import pandas as pd
from config import FEATURE_NAMES, SCALER_PATH, KMEANS_PATH


app = FastAPI()

SEGMENT_NAMES = {
    0: "mid 50s, middle income, middle spender",
    1: "mid 20s, middle income, middle spender",
    2: "late 30s to early 40s, high income, low spender",
    3: "late 20s to early 30s, high income, high spender",
    4: "mid 20s, low income, high spender",
    5: "mid 40s, low income, low spender"
}


class Customer(BaseModel):
    age: int = Field(ge = 18, le = 80)
    income: float = Field(ge = 15, le = 137)
    spending_score: int = Field(ge = 1, le = 100)

kmeans_model = joblib.load(KMEANS_PATH)
scaler_model = joblib.load(SCALER_PATH)

class SegmentResponse(BaseModel):
    segment: int
    segment_name: str

@app.post("/predict", response_model=SegmentResponse)
def predict(customer: Customer):
    customer_data = pd.DataFrame([[customer.age, customer.income, customer.spending_score]], columns=FEATURE_NAMES)
    customer_data_scaled = scaler_model.transform(customer_data)
    cluster_pred = kmeans_model.predict(customer_data_scaled)
    cluster = int(cluster_pred[0])
    return {"segment": cluster, "segment_name": SEGMENT_NAMES[cluster]}

