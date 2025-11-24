import requests
import pandas as pd
import io


#  --- 1. READ DOWNLOADED CSV FILES
def get_csv(filepath: str) -> pd.DataFrame:
    """
    Converts a downloaded CSV file into a pandas DataFrame.

    Args:
        filepath: The filepath of the CSV file for Apple Watch and Fitbit data in the local machine.

    Returns: 
        pandas DataFrame (df) or None
    """
    print(f"--- Loading {filepath} into DataFrame ---")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully")
        return df
    
    except Exception as e:
        print(f"Error loading into DataFrame: {e}")
        return None


# --- 2. DOWNLOAD DATA WITH API
def get_chronic_data(url: str) -> pd.DataFrame:
    """
    Downloading data using a RESTful web API and converting it into a pandas DataFrame.

    Args: 
        url: The URL for the API

    Returns:
        pandas DataFrame (df) or None
    """
    print(f"--- Downloading {url} ---")
    try: 
        response = requests.get(url)
        response.raise_for_status()

        print("loading into DataFrame...")
        df = pd.read_csv(io.BytesIO(response.content))
        print("Data loaded successfully")
        return df
    
    except Exception as e:
        print(f"Error downloading or loading data: {e}")
        return None

