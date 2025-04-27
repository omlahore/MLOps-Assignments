import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class HouseFeatures(BaseModel):
    """
    Wrapper for incoming JSON prediction requests.
    """
    features: dict = Field(..., description="Dictionary of feature names and values")

class HousePricePredictor:
    """
    Preprocesses data and trains a linear regression model on the House Prices dataset.
    """
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at {data_path}")

        df = pd.read_csv(data_path)
        y = df['SalePrice']
        X = df.drop(['Id', 'SalePrice'], axis=1)

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()

    def predict(self, input_data: dict) -> float:
        missing = set(self.feature_names) - set(input_data.keys())
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        df = pd.DataFrame([input_data], columns=self.feature_names)
        return float(self.model.predict(df)[0])

# FastAPI app
app = FastAPI(
    title="House Price Predictor",
    description="Predict house sale prices with a linear regression model",
    version="1.0"
)

predictor = None
DATA_PATH = os.getenv('HOUSE_DATA_PATH', 'train.csv')

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = HousePricePredictor(DATA_PATH)
        print(f"Model loaded with data from {DATA_PATH}")
    except Exception as e:
        # Reraise so uvicorn fails early with useful error
        raise RuntimeError(f"Failed to load HousePricePredictor: {e}")

@app.post("/predict")
def predict_price(house: HouseFeatures):
    """
    Predict house price for provided features.

    Request body: { "features": { feature_name: value, ... } }
    Response: { "predicted_price": float }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        price = predictor.predict(house.features)
        return {"predicted_price": price}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

# To run:
# uvicorn house_price_api:app --reload
