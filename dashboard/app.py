import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import prepare_country_data
from models.prophet_model import train_prophet

st.markdown(
    """
    <div style="
        background-color:#0E1117;
        padding:15px;
        border-radius:10px;
        text-align:center;
    ">
        <h1 style="color:white;">🌍 COVID-19 AI Forecast Dashboard</h1>
        <p style="color:lightgray;">
        Machine Learning based pandemic analysis and forecasting platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.info(
    "📊 Explore global COVID trends, analyze historical data, and predict future cases using AI forecasting models."
)

st.title("COVID-19 Forecast Dashboard")
cases, deaths = load_data()
world = pd.DataFrame({
    "Country": cases.columns[1:]
})

case_values = []

for country in world["Country"]:
    case_values.append(pd.to_numeric(cases[country][1:]).sum())

world["Cases"] = case_values
world["Cases Range"] = pd.cut(
    world["Cases"],
    [-150000,50000,200000,800000,1500000,15000000,50000000,200000000],
    labels=[
        "U50K",
        "50Kto200K",
        "200Kto800K",
        "800Kto1.5M",
        "1.5Mto15M",
        "15Mto50M",
        "50M+"
    ]
)
continent = pd.read_csv("data/continents2.csv")
continent["name"] = continent["name"].str.upper()
alpha = []

for c in world["Country"].str.upper():
    if len(continent[continent["name"] == c]["alpha-3"].values) == 0:
        alpha.append(None)
    else:
        alpha.append(continent[continent["name"] == c]["alpha-3"].values[0])

world["Alpha3"] = alpha
countries = list(cases.columns[1:])
country = st.selectbox("Select Country", countries)
df = prepare_country_data(cases, country)
model, forecast = train_prophet(df)


st.subheader("Forecast for next 30 days")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("Historical Cases")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["ds"], df["y"], linewidth=2)
ax.set_xlabel("Date", fontsize=8)
ax.set_ylabel("Cases", fontsize=10)
ax.tick_params(axis='x', labelsize=8, rotation=45)
ax.tick_params(axis='y', labelsize=8)
fig.autofmt_xdate()
st.pyplot(fig)

st.subheader("Global COVID Distribution")
fig = px.choropleth(
    world.dropna(),
    locations="Alpha3",
    color="Cases Range",
    projection="mercator"
)
st.plotly_chart(fig)

st.subheader("Top 10 Countries by Total Cases")
top10 = world.sort_values("Cases", ascending=False).head(10)
fig_top10 = px.bar(
    top10,
    x="Country",
    y="Cases",
    color="Cases",
    title="Top 10 Countries with Highest COVID Cases"
)
st.plotly_chart(fig_top10)

st.subheader("COVID Growth Rate Indicator")
today_cases = df["y"].iloc[-1]
yesterday_cases = df["y"].iloc[-2]
growth_rate = (today_cases - yesterday_cases) / yesterday_cases * 100
st.metric(
    label="Daily Growth Rate (%)",
    value=f"{growth_rate:.2f}%"
)

st.subheader("Animated Global COVID Spread")
daily_world = []
for i in range(1, len(cases)):
    date = cases["Country/Region"][i]
    for country in cases.columns[1:]:
        daily_world.append({
            "Date": date,
            "Country": country,
            "Cases": pd.to_numeric(cases[country][i])
        })

daily_world_df = pd.DataFrame(daily_world)

fig_anim = px.choropleth(
    daily_world_df,
    locations="Country",
    locationmode="country names",
    color="Cases",
    animation_frame="Date",
    projection="mercator"
)
st.plotly_chart(fig_anim)

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center">
        <p>© 2026 COVID AI Forecast Dashboard</p>
        <p>Developed by <b>Karthikeyan</b> | AI & Data Science Project</p>
        <p>
        <a href="https://www.linkedin.com/in/karthikeyan-selvamani" target="_blank">
        🔗 Connect on LinkedIn
        </a>
        </p>
        <p>
        Built with ❤️ using Python, Streamlit, Prophet, TensorFlow
        </p>
    </div>
    """,
    unsafe_allow_html=True
)