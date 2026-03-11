import pandas as pd

def load_data():

    cases = pd.read_csv("data/CONVENIENT_global_confirmed_cases.csv")
    deaths = pd.read_csv("data/CONVENIENT_global_deaths.csv")

    return cases, deaths