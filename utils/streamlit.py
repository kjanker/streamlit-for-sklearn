"""
Helper script to handle some boilerplate streamlit code.
"""
import altair as alt
import pandas as pd
import streamlit as st


def model_settings(settings: dict, columns: int = 3):
    cols = st.columns(columns)
    kwargs = {}
    for index, (key, value) in enumerate(settings.items()):
        col = cols[index % columns]
        if isinstance(value, int):
            kwargs[key] = col.number_input(key, step=1, format="%i", value=value)
        elif isinstance(value, float):
            kwargs[key] = col.number_input(key, step=0.1, format="%.3f", value=value)
        elif isinstance(value, list):
            kwargs[key] = col.selectbox(key, options=value)
        else:
            col.write(key)
    return kwargs


def axis_selector(columns: list):
    left, right = st.columns(2)
    x_axis_col = left.selectbox("X axis:", options=columns, index=0)
    y_axis_col = right.selectbox(
        "Y axis:", options=columns, index=min(len(columns) - 1, 1)
    )
    return x_axis_col, y_axis_col


def plot_dataset(df: pd.DataFrame, x: str, y: str):
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(x, scale=alt.Scale(zero=False, padding=1)),
            y=alt.Y(y, scale=alt.Scale(zero=False, padding=1)),
            color="cluster",
            tooltip=list(df.columns),
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
