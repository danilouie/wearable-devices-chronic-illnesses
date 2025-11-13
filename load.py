import requests
import pandas as pd

def get_csv(filepath):
    return pd.read_csv(filepath)

def get_chronic_data():
    url = "https://data.cdc.gov/api/views/hksd-2xuw/rows.csv?accessType=DOWNLOAD"
    output_filename = "cdc_chronic_disease_indicators.csv"
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_filename, "wb") as f:
        f.write(response.content)

    return output_filename


