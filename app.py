"""
Streamlit app and main entry point. Run with `streamlit run app.py`.
"""
from warnings import catch_warnings

import pandas as pd
import streamlit as st

import utils.streamlit as stx
from utils.code_builder import make_code_block
from utils.sklearn import datasets, models

### Title ######################################################################
title, logo1, logo2 = st.columns([6, 2, 2])
title.header("Clustering with sklearn")
logo1.image(
    "https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png",
    use_column_width=True,
)
logo2.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1280px-Scikit_learn_logo_small.svg.png",
    use_column_width=True,
)

### Sample data ################################################################
st.subheader("Sample data")

dataset_key = st.selectbox("Choose a dataset:", options=datasets.keys())

with st.spinner("Loading dataset..."):
    dataset = datasets[dataset_key]()

with st.expander("Show description"):
    st.write(dataset["DESCR"])

df = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
df["target"] = dataset["target"]

st.dataframe(df)

### Define model ###############################################################
st.subheader("Define model")

model_key = st.selectbox("Select a model:", options=models.keys())
use_adv = st.checkbox("Advanced arguments", value=False)

model_settings = models[model_key]["kwargs"]
if use_adv:
    model_settings = {**model_settings, **models[model_key]["kwargs_adv"]}

model_kwargs = stx.model_settings(model_settings)
model = models[model_key]["class"](**model_kwargs)

### Fit model ##################################################################
st.subheader("Fit model")

model_is_fitted = False

data_columns = st.multiselect(
    "Feature selection:",
    options=df.columns,
    default=list(df.columns),
)
data = df[data_columns].values
if data.shape[1] == 0:
    st.warning("Please select some data first.")
else:
    with st.spinner("Fitting the model..."):
        with catch_warnings(record=True) as warnings:
            try:
                model.fit(data)
            except Exception as error:
                st.error(
                    f"**Incorrect model arguments:** {error} ({error.__class__.__name__})"
                )
            else:
                for warning in warnings:
                    st.warning(f"**{warning.category.__name__}:** {warning.message}")
                st.success("**Success:** Model is fitted to the data.")
                model_is_fitted = True

if model_is_fitted:
    df["cluster"] = model.labels_.astype(str)

### Plot results ###############################################################
st.subheader("Plot results")

if not model_is_fitted:
    st.warning("Model needs to be fitted first.")
    st.stop()

plot_x_axis, plot_y_axis = stx.axis_selector(df.columns)
stx.plot_dataset(df, plot_x_axis, plot_y_axis)

### Python code ################################################################
st.subheader("Python code")

code = make_code_block(
    model=model,
    data_callable=datasets[dataset_key],
    feature_cols=data_columns,
    plot_x_axis=plot_x_axis,
    plot_y_axis=plot_y_axis,
)
st.code(code, language="python")
