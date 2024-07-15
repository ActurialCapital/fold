from functools import partial

import pandas as pd

from fold import *

seed = 42

index = pd.date_range("2020-01-01", "2020-02-01", inclusive="left")

assert_index_equal = partial(
    pd.testing.assert_index_equal,
    rtol=1e-06,
    atol=0
)


def test_plot():
    fig = BaseModel(index, [0.5]).plot()
    # Check that the output is a plotly.graph_objs._figure.Figure instance
    assert fig.__class__.__name__ == 'FigureWidget', "Output is not a Plotly FigureWidget"

    # Verify the data and layout of the figure
    assert fig.data[0].type == 'heatmap', "First trace is not a heatmap plot"
    # Index
    assert_index_equal(pd.DatetimeIndex(fig.data[0].x), index)
