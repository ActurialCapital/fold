import abc
from dataclasses import dataclass
import importlib
import typing as tp

from fold.utils import checks


class BasePeriod(abc.ABC):
    """
    Abstract base class for defining period-related operations.

    Parameters
    ----------
    period : tp.Any
        The period object representing the time period.
    index : range
        The index or range of the period.
    allow_relative : bool, optional, default=False
        Whether relative periods are allowed.
    allow_zero_len : bool, optional, default=False
        Whether a period of zero length is allowed.
    range_format : str, optional, default="slice_or_any"
        The format of the range, specifying slice or other formats.

    """
    @abc.abstractmethod
    def __init__(
        self,
        period: tp.Any,
        index: range,
        allow_relative: tp.Optional[bool] = False,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = "slice_or_any",
    ):
        pass

    @abc.abstractmethod
    def bounds(self, *args, **kwargs):
        """
        Abstract method to calculate the bounds of the period.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass

    @abc.abstractmethod
    def span(self, *args, **kwargs):
        """
        Abstract method to calculate the span of the period.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass

    @abc.abstractmethod
    def coverage(self, *args, **kwargs):
        """
        Abstract method to determine the coverage of the period.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass

    @abc.abstractmethod
    def __post_init__(self):
        """
        Abstract method for post-initialization operations.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass


class BaseMetadata:
    """
    Base class for storing metadata related to periods.

    Parameters
    ----------
    period : tp.Any
        The period object representing the time period.
    index : range
        The index or range of the period.
    range_format : str, optional
        The format of the range, specifying slice or other formats.
    start : int, optional
        The start of the period.
    stop : int, optional
        The stop of the period.
    length : int, optional
        The length of the period.

    """

    def __init__(
        self,
        period: tp.Any,
        index: range,
        range_format: tp.Optional[str] = None,
        start: tp.Optional[int] = None,
        stop: tp.Optional[int] = None,
        length: tp.Optional[int] = None,

    ):
        self.period = period
        self.index = index
        self.range_format = range_format
        self.start = start
        self.stop = stop
        self.length = length

    @property
    def output(self) -> tp.Dict[str, tp.Any]:
        """
        Returns a dictionary representation of metadata.

        Returns
        -------
        dict
            Dictionary containing period metadata.
        """
        return dict(
            period=self.period,
            index=self.index,
            range_format=self.range_format,
            start=self.start,
            stop=self.stop,
            length=self.length
        )


@dataclass(frozen=True)
class BaseTool(abc.ABC):
    """
    Abstract Class for substituting templates.

    Attributes
    ----------
    template : Any
        Template to be processed.
    context : tp.Dict[str, tp.Any], optional
        Context mapping, by default None.
    context_merge_kwargs : tp.Dict[str, tp.Any], optional
        Keyword arguments passed to `.merge_dicts`, by default None.
    eval_id : Optional[str], optional
        One or more identifiers at which to evaluate this instance,
        by default None.

    Methods
    -------
    meets_eval_id
        Return whether the evaluation id of the instance meets the global 
        evaluation id.
    resolve_context
        Resolve `.context`. Merges `context` in `.template`, `.context`, and 
        `context`. Automatically appends `eval_id`, `np` and `pd`.
    substitute
        Abstract method to substitute the template `.template` 
        using the context from merging `.context` and `context`.
    """

    template: tp.Any
    context: tp.Dict[str, tp.Any] = None
    strict: bool = None
    context_merge_kwargs: tp.Dict[str, tp.Any] = None
    eval_id: str = None

    def meets_eval_id(self, eval_id: tp.Optional[tp.Hashable]) -> bool:
        """
        Return whether the evaluation id of the instance meets the global 
        evaluation id.

        Parameters
        ----------
        eval_id : Optional[Hashable]
            Global evaluation identifier.

        Returns
        -------
        bool
            True if the evaluation id of the instance matches the given
            `eval_id`, False otherwise.

        """
        if self.eval_id is not None and eval_id is not None:
            if checks.is_complex_sequence(self.eval_id):
                if eval_id not in self.eval_id:
                    return False
            else:
                if eval_id != self.eval_id:
                    return False
        return True

    def resolve_context(
        self,
        context: tp.Optional[tp.Dict[str, tp.Any]] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Dict[str, tp.Any]:
        """
        Resolve `BaseTool.context`.

        Parameters
        ----------
        context : tp.Dict[str, tp.Any], optional
            Additional context mapping to merge, by default None.
        eval_id : Optional[Hashable], optional
            Evaluation identifier to append to the context, by default None.

        Returns
        -------
        Kwargs
            Merged context dictionary including `.template`, `.context`, 
            and `context` with appended `eval_id`, `np` (NumPy), `pd` (Pandas).

        """
        _self_context = self.context or {}
        context = context or {}

        context_merge_kwargs = self.context_merge_kwargs
        if context_merge_kwargs is None:
            context_merge_kwargs = {}

        new_context = {**_self_context, **context, **context_merge_kwargs}

        if "context" not in new_context:
            new_context["context"] = dict(new_context)

        if "eval_id" not in new_context:
            new_context["eval_id"] = eval_id

        package_shortcut_config = dict(
            pd="pandas",
            np="numpy",
            nb="numba"
        )
        for k, v in package_shortcut_config.items():
            if k not in new_context:
                try:
                    new_context[k] = importlib.import_module(v)
                except ImportError:
                    pass
        return new_context

    def resolve_strict(self, strict: tp.Optional[bool] = None) -> bool:
        """Resolve strict`."""
        if strict is None:
            strict = self.strict

        if strict is None:
            strict = True

        return strict

    def substitute(
        self,
        context: tp.Optional[tp.Dict[str, tp.Any]] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """
        Abstract method to substitute the template `.template` 
        using the merged context.

        Parameters
        ----------
        context : tp.Dict[str, tp.Any], optional
            Additional context mapping to merge, by default None.
        eval_id : Optional[Hashable], optional
            Evaluation identifier to append to the context, by default None.

        Returns
        -------
        Any
            Result of the template substitution.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the derived class.
        """
        raise NotImplementedError


class BaseTemplate(abc.ABC):
    """
    Abstract Class for period templates.
    """
    pass
