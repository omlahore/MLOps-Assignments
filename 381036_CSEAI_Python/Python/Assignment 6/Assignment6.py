import os
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# File: Assignment6.py

class IrisFilterRequest(BaseModel):
    """
    Request model for filtering iris dataset.
    """
    species: List[str] = Field(..., description="List of species to include (e.g., ['setosa', 'versicolor'])")
    features: Optional[List[str]] = Field(None, description="List of numeric features to visualize. If None, defaults will be used.")

class IrisFilterResponse(BaseModel):
    filtered_count: int
    plots: List[str]
    sample: List[dict]

class IrisDataFilter:
    """
    Filter iris data by species and create feature distribution visualizations.
    """
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Iris data not found at {data_path}")
        self.df = pd.read_csv(data_path)

    def filter_by_species(self, species_list: List[str]) -> pd.DataFrame:
        """
        Return subset of data where 'species' is in the provided list.
        """
        return self.df[self.df['species'].isin(species_list)]

    def plot_feature_distribution(self, df: pd.DataFrame, feature: str, output_dir: str) -> str:
        """
        Create and save a histogram for the given feature in df.
        Returns the filepath of the saved image.
        """
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

# Initialize FastAPI
app = FastAPI(
    title="Iris Data Filter",
    description="Filter Iris dataset by species and visualize feature distributions",
    version="1.0"
)

# Load data filter instance on import
DATA_PATH = os.getenv('IRIS_DATA_PATH', 'iris.csv')
try:
    iris_filter = IrisDataFilter(DATA_PATH)
    print(f"Loaded iris data from {DATA_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load iris data: {e}")

@app.post("/filter", response_model=IrisFilterResponse)
def filter_iris(req: IrisFilterRequest):
    """
    Filter iris data by species and return sample and distribution plots.
    """
    # Validate request
    if not req.species:
        raise HTTPException(status_code=422, detail="At least one species must be provided.")

    # Perform filtering
    df_filtered = iris_filter.filter_by_species(req.species)
    count = len(df_filtered)

    # Choose features to plot
    default_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    feats = req.features or default_features

    # Generate plots
    output_dir = 'iris_plots'
    plot_paths = []
    for feat in feats:
        try:
            path = iris_filter.plot_feature_distribution(df_filtered, feat, output_dir)
            plot_paths.append(path)
        except ValueError:
            continue

    # Return a small sample (first 10 rows)
    sample = df_filtered.head(10).to_dict(orient='records')
    return IrisFilterResponse(filtered_count=count, plots=plot_paths, sample=sample)

# To run:
#   uvicorn Assignment6:app --reload
