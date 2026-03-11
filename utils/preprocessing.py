import pandas as pd


def prepare_country_data(df_cases, country):
    dates = pd.to_datetime(df_cases["Country/Region"][1:])
    cases = pd.to_numeric(df_cases[country][1:])
    df = pd.DataFrame()
    df["ds"] = dates
    df["y"] = cases.values
    return df