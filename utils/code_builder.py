"""
Helper functions to create sample code.
"""


def get_import_path(object):
    return ".".join([a for a in object.__module__.split(".") if a[0] != "_"])


def make_code_block(model, data_callable, feature_cols, plot_x_axis, plot_y_axis):
    return f"""
import altair as alt
import pandas as pd
from {get_import_path(model)} import {model.__class__.__name__}
from {get_import_path(data_callable)} import {data_callable.__name__}

# load and prepare the data
dataset = {data_callable.__name__}()
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
df['target'] = dataset['target']
data = df[{feature_cols}].values

# create and fit the model
model = {model}
model.fit(data)

# get the results
df['cluster'] = model.labels_.astype(str)

# plot data and results
alt.Chart(df).mark_circle(size=60).encode(
    x=alt.X('{plot_x_axis}', scale=alt.Scale(zero=False, padding=1)),
    y=alt.Y('{plot_y_axis}', scale=alt.Scale(zero=False, padding=1)),
    color='cluster',
    tooltip=list(df.columns),
).interactive()
    """
