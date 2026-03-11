import plotly.express as px


def world_map(world):
    fig = px.choropleth(
        world,
        locations="Alpha3",
        color="Cases Range",
        projection="mercator"
    )

    return fig