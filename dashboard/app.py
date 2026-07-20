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


st.set_page_config(
    page_title="COVID-19 AI Forecast",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(-45deg, #0b0e1a, #141b2d, #1a1f3a, #0f1629);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 30px 60px -20px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.06);
        padding: 1.75rem 1.75rem;
        transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        margin-bottom: 1.25rem;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 40px 80px -20px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.10);
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #f0f4ff !important;
    }

    h1 {
        font-size: 2.6rem !important;
        background: linear-gradient(135deg, #f0f4ff 0%, #a0b8ff 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        display: inline-block;
        padding-bottom: 0.1rem;
    }

    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 1.2rem 1.2rem;
        box-shadow: 0 8px 24px -8px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }

    div[data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }

    div[data-testid="metric-container"] label {
        color: #a8b8e8 !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em;
    }

    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: #f0f4ff !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 14px !important;
        color: #f0f4ff !important;
        padding: 0.6rem 1rem !important;
        font-weight: 400;
        transition: all 0.3s ease;
    }

    .stSelectbox div[data-baseweb="select"] > div:focus {
        border-color: #4a6cf7 !important;
        box-shadow: 0 0 0 3px rgba(74, 108, 247, 0.25) !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #4a6cf7, #6a3de8) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.6rem 1.8rem !important;
        box-shadow: 0 8px 28px -6px rgba(74, 108, 247, 0.45) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }

    .stButton button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 16px 40px -8px rgba(74, 108, 247, 0.65) !important;
    }

    .stPlotlyChart, .stPyplot {
        background: rgba(255,255,255,0.02);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.05);
        overflow: hidden;
        backdrop-filter: blur(8px);
        padding: 0.5rem;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 2rem 0;
    }

    .stAlert {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        color: #e0e8ff !important;
    }
</style>
""", unsafe_allow_html=True)


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


st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.25rem;">
        <span style="font-size: 2.8rem;">🦠</span>
        <h1 style="display: inline-block;">COVID-19 AI Forecast Dashboard</h1>
        <span class="shiny-badge" style="margin-left: 0.5rem;">Prophet</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.info(
    "📊 Explore global COVID trends, analyze historical data, and predict future cases using AI forecasting models.",
    icon="ℹ️"
)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_sel, col_metric = st.columns([2, 1])
    with col_sel:
        country = st.selectbox("Select a Country", countries)
    with col_metric:
        pass
    st.markdown('</div>', unsafe_allow_html=True)

df = prepare_country_data(cases, country)
model, forecast = train_prophet(df)

today_cases = df["y"].iloc[-1]
yesterday_cases = df["y"].iloc[-2]
growth_rate = (today_cases - yesterday_cases) / yesterday_cases * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="📈 Total Cases (selected)",
        value=f"{df['y'].iloc[-1]:,.0f}",
        delta=f"{df['y'].iloc[-1] - df['y'].iloc[-2]:,.0f} from yesterday"
    )
with col2:
    st.metric(
        label="📊 Daily Growth Rate",
        value=f"{growth_rate:.2f}%",
        delta="based on last 2 days"
    )
with col3:
    st.metric(
        label="🔮 Forecast (30 days)",
        value=f"{forecast['yhat'].iloc[-1]:,.0f}",
        delta="predicted"
    )

st.markdown("---")

# ── Forecast Plot ──
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Forecast for Next 30 Days")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Historical Cases ──
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📉 Historical Cases")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["ds"], df["y"], linewidth=2, color="#4a6cf7")
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Cases", fontsize=10)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8)
    fig.autofmt_xdate()
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.spines['bottom'].set_color('#445577')
    ax.spines['left'].set_color('#445577')
    ax.xaxis.label.set_color('#c0d0ff')
    ax.yaxis.label.set_color('#c0d0ff')
    ax.tick_params(colors='#c0d0ff')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Two columns: World Map + Top 10 ──
col_map, col_top = st.columns([3, 2])

with col_map:
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🌍 Global COVID Distribution")
        fig = px.choropleth(
            world.dropna(),
            locations="Alpha3",
            color="Cases Range",
            projection="mercator",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#c0d0ff")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_top:
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🏆 Top 10 Countries")
        top10 = world.sort_values("Cases", ascending=False).head(10)
        fig_top10 = px.bar(
            top10,
            x="Country",
            y="Cases",
            color="Cases",
            color_continuous_scale=px.colors.sequential.Blues_r,
            title="Top 10 Countries with Highest COVID Cases"
        )
        fig_top10.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0d0ff"),
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig_top10, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎞️ Animated Global COVID Spread")

    daily_world = []
    for i in range(1, len(cases)):
        date = cases["Country/Region"][i]
        daily_world.extend([
            {
                "Date": date,
                "Country": country,
                "Cases": pd.to_numeric(cases[country][i])
            }
            for country in cases.columns[1:]
        ])

    daily_world_df = pd.DataFrame(daily_world)

    fig_anim = px.choropleth(
        daily_world_df,
        locations="Country",
        locationmode="country names",
        color="Cases",
        animation_frame="Date",
        projection="mercator",
        color_continuous_scale=px.colors.sequential.Reds_r
    )
    fig_anim.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#c0d0ff")
    )
    st.plotly_chart(fig_anim, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color: #556688; padding: 1rem 0;">
        <p>© 2026 COVID AI Forecast Dashboard</p>
        <p>Developed by <b style="color:#a8b8e8;">Karthikeyan</b> | AI & Data Science Project</p>
        <p>
        <a href="https://www.linkedin.com/in/karthikeyan-selvamani" target="_blank" style="color:#4a6cf7; text-decoration:none;">
        🔗 Connect on LinkedIn
        </a>
        </p>
        <p style="font-size:0.85rem; color:#445577;">
        Built with ❤️ using Python, Streamlit, Prophet, Plotly
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
