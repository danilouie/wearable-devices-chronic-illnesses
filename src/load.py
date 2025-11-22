import requests
import pandas as pd
import io

# TODO: add try and except error

#  --- 1. READ DOWNLOADED CSV FILES
def get_csv(filepath: str) -> pd.DataFrame:
    """
    Converts a downloaded CSV file into a pandas DataFrame.

    :param filepath: The filepath of the CSV file in the local machine.
    :return: pandas DataFrame or None
    """
    print(f"---Loading {filepath} into DataFrame ---")
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

    :param url: The URL for the API
    :return: pandas DataFrame or None
    """
    print(f"---Downloading {url} ---")
    try: 
        response = requests.get(url)
        response.raise_for_status()

        print("floading into DataFrame...")
        df = pd.read_csv(io.BytesIO(response.content))
        print("Data loaded successfully")
        return df
    
    except Exception as e:
        print(f"Error downloading or loading data: {e}")
        return None

