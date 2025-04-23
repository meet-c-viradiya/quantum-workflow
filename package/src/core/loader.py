# File: /quantum-workflow/quantum-workflow/src/core/loader.py

from typing import Any, Dict
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load workflow data from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the workflow data."""
    # Example preprocessing steps
    df.dropna(inplace=True)  # Remove rows with missing values
    df.reset_index(drop=True, inplace=True)
    return df

def convert_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert DataFrame to a dictionary for easier access."""
    return df.to_dict(orient='records')