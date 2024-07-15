import abc
from dataclasses import dataclass, field
import typing as tp

import numpy as np

import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure, BaseTraceType


class BasePlotter(abc.ABC):
    def __init__(self, figure: BaseFigure, traces: tp.Tuple[BaseTraceType, ...]):
        """Base trace updating class."""
        self._figure = figure
        self._traces = traces

    @property
    def figure(self) -> BaseFigure:
        """Figure."""
        return self._figure

    @property
    def traces(self) -> tp.Tuple[BaseTraceType, ...]:
        """Traces to update."""
        return self._traces

    @classmethod
    @abc.abstractmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        data: np.ndarray,
        *args,
        **kwargs
    ):
        """Update one trace."""
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Update all traces using new data."""
        pass


@dataclass
class Layout:
    width: int = 700
    height: int = 350
    margin: dict = field(
        default_factory=lambda: dict(t=30, b=30, l=30, r=30)
    )
    legend: dict = field(
        default_factory=lambda: dict(
            x=1,
            y=1.02, orientation="h",
            yanchor="bottom",
            xanchor="right",
            traceorder="normal"
        )
    )
    # Scikit-learn colors
    colorway: list = field(
        default_factory=lambda: [
            "#007AB8",
            "#F7931E",
            "#505050"
        ]
    )


@dataclass
class Trace:
    hoverongaps: bool = False
    showscale: bool = False
    showlegend: bool = True


class Heatmap(BasePlotter):
    """
    Create a heatmap plot.

    Parameters
    ----------
    data : array_like, optional
        Data in any format that can be converted to NumPy.
    is_x_category : bool, optional
        Whether X-axis is a categorical axis. Default is False.
    is_y_category : bool, optional
        Whether Y-axis is a categorical axis. Default is False.
    trace_kwargs : dict, optional
        Keyword arguments passed to `plotly.graph_objects.Heatmap`.
    add_trace_kwargs : dict, optional
        Keyword arguments passed to `add_trace`.
    fig : Figure or FigureWidget, optional
        Figure to add traces to.
    **layout_kwargs : dict
        Keyword arguments for layout.
    """

    def __init__(
        self,
        data: tp.Optional[np.ndarray],
        is_x_category: tp.Optional[bool] = False,
        is_y_category: tp.Optional[bool] = False,
        trace_kwargs: tp.Dict[str, tp.Any] = None,
        add_trace_kwargs: tp.Dict[str, tp.Any] = None,
        figure_kwargs: tp.Dict[str, tp.Any] = None,
        fig: tp.Optional[BaseFigure] = None,
        **layout_kwargs
    ):
        trace = go.Heatmap(**trace_kwargs)

        if data is not None:
            self.update_trace(trace, data)
            
        fig.add_trace(trace, **add_trace_kwargs)

        axis_kwargs = dict()
        if is_x_category:
            if fig.data[-1]["xaxis"] is not None:
                axis_kwargs["xaxis" + fig.data[-1]["xaxis"][1:]] = dict(type="category")
            else:
                axis_kwargs["xaxis"] = dict(type="category")

        if is_y_category:
            if fig.data[-1]["yaxis"] is not None:
                axis_kwargs["yaxis" + fig.data[-1]["yaxis"][1:]] = dict(type="category")
            else:
                axis_kwargs["yaxis"] = dict(type="category")

        fig.update_layout(**axis_kwargs)
        fig.update_layout(**layout_kwargs)

        super().__init__(fig, (fig.data[-1],))

    @classmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        data: np.ndarray,
        *args,
        **kwargs
    ):
        trace.z = data

    def update(self, data: np.ndarray):
        with self.fig.batch_update():
            self.update_trace(self.traces[0], data)
