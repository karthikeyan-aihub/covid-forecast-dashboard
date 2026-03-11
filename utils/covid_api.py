import requests
import pandas as pd

#for API data fetching and processing real time data from disease.sh API
def fetch_global_data():
    url = "https://disease.sh/v3/covid-19/historical/all?lastdays=all"
    data = requests.get(url).json()
    cases = data["cases"]
    df = pd.DataFrame(list(cases.items()), columns=["ds","y"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df