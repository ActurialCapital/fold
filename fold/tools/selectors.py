from dataclasses import dataclass
import typing as tp


@dataclass(frozen=True)
class SelectPosition:
    """
    Class that represents a selection by position.

    Parameters
    ----------
    value : tp.Any
        Selection of one or more positions.

    """
    value: tp.Any


@dataclass(frozen=True)
class SelectLabel:
    """
    Class that represents a selection by label.

    Parameters
    ----------
    value : tp.Any
        Selection of one or more labels.

    """
    value: tp.Any
