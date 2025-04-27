import os
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# File: Assignment3.py

class WineFilterRequest(BaseModel):
    min_quality: int = Field(..., description="Minimum wine quality (inclusive)")
    max_quality: Optional[int] = Field(None, description="Maximum wine quality (inclusive). If None, no upper bound.")
    features: Optional[List[str]] = Field(None, description="List of feature names to visualize. If None, default features will be used.")

class WineFilterResponse(BaseModel):
    filtered_count: int
    plots: List[str]
    sample: List[dict]

class WineDataFilter:
    """
    Filter wine data by quality and create feature distribution visualizations.
    """
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Wine data not found at {data_path}")
        self.df = pd.read_csv(data_path, sep=';')

    def filter_by_quality(self, min_q: int, max_q: Optional[int] = None) -> pd.DataFrame:
        if max_q is not None:
            return self.df[(self.df['quality'] >= min_q) & (self.df['quality'] <= max_q)]
        return self.df[self.df['quality'] >= min_q]

    def plot_feature_distribution(self, df: pd.DataFrame, feature: str, output_dir: str) -> str:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in data")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"dist_{feature}.png")
        plt.figure(figsize=(8, 6))
        df[feature].hist(bins=20)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return filepath

app = FastAPI(
    title="Wine Quality Filter",
    description="Filter red wine data by quality and visualize feature distributions",
    version="1.0"
)

# Load data filter instance on import
DATA_PATH = os.getenv('WINE_DATA_PATH', 'winequality-red.csv')
try:
    wine_filter = WineDataFilter(DATA_PATH)
    print(f"Loaded wine data from {DATA_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load wine data: {e}")

@app.post("/filter", response_model=WineFilterResponse)
def filter_wine(req: WineFilterRequest):
    if req.min_quality < 0:
        raise HTTPException(status_code=422, detail="min_quality must be non-negative")
    if req.max_quality is not None and req.max_quality < req.min_quality:
        raise HTTPException(status_code=422, detail="max_quality must be >= min_quality")

    df_filtered = wine_filter.filter_by_quality(req.min_quality, req.max_quality)
    count = len(df_filtered)

    default_feats = ['alcohol', 'sulphates', 'citric acid']
    feats = req.features or default_feats

    output_dir = 'wine_plots'
    plot_paths = []
    for feat in feats:
        try:
            path = wine_filter.plot_feature_distribution(df_filtered, feat, output_dir)
            plot_paths.append(path)
        except ValueError:
            continue

    sample = df_filtered.head(10).to_dict(orient='records')
    return WineFilterResponse(filtered_count=count, plots=plot_paths, sample=sample)

# To run:
#   uvicorn Assignment3:app --reload
