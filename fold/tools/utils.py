import ast
import inspect
from string import Template
from copy import copy
import typing as tp

from fold.tools.base import BaseTool
from fold.utils import checks


class Replace(BaseTool):
    """
    Template string to be replaced with the respective value from `context`.

    Methods
    -------
    substitute
        Replace a key.
    """

    def substitute(
        self,
        context:  tp.Optional[tp.Dict[str, tp.Any]] = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """Substitute parts of `Sub.template` as a regular template."""
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return Template(self.template).substitute(context)
        except KeyError as e:
            if strict:
                raise e
        return self


class Key(BaseTool):
    """
    Template string to be replaced with the respective value from `context`.

    Methods
    -------
    substitute
        Replace a key.
    """

    def substitute(
        self,
        context: tp.Optional[tp.Dict[str, tp.Any]] = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """
        Replace `Rep.template` as a key.

        Parameters
        ----------
        context : tp.Dict[str, tp.Any], optional
            Additional context mapping to merge, by default None.
        eval_id : Optional[Hashable], optional
            Evaluation identifier to append to the context, by default None.

        Returns
        -------
        Any
            Value corresponding to the template key in the context.
        """
        if not self.meets_eval_id(eval_id):
            return self

        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return context[self.template]
        except KeyError as e:
            if strict:
                raise e
        return self


class Lambda(BaseTool):
    """
    Template expression to be evaluated using with `context` used as locals.

    Methods
    -------
    multiline_eval
        Evaluate several lines of input, returning the result of the last line.

    substitute
        Evaluate an expression.
    """

    @staticmethod
    def multiline_eval(
        expr: str,
        context: tp.Optional[tp.Dict[str, tp.Any]] = None
    ) -> tp.Any:
        """
        Evaluate several lines of input, returning the result of the last line.

        Parameters
        ----------
        expr : str
            Multiline expression to evaluate.
        context : tp.Dict[str, tp.Any], optional
            Local context for evaluation, by default None.

        Returns
        -------
        Any
            Result of the evaluation.
        """
        context = context or {}
        tree = ast.parse(inspect.cleandoc(expr))
        eval_expr = ast.Expression(tree.body[-1].value)
        exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
        exec(compile(exec_expr, "file", "exec"), context)
        return eval(compile(eval_expr, "file", "eval"), context)

    def substitute(
        self,
        context: tp.Optional[tp.Dict[str, tp.Any]] = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """
        Evaluate `RepEval.template` as an expression.

        Parameters
        ----------
        context : tp.Dict[str, tp.Any], optional
            Additional context mapping to merge, by default None.
        eval_id : Optional[Hashable], optional
            Evaluation identifier to append to the context, by default None.

        Returns
        -------
        Any
            Result of the expression evaluation.
        """
        if not self.meets_eval_id(eval_id):
            return self

        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return self.multiline_eval(self.template, context)
        except NameError as e:
            if strict:
                raise e
        return self


class Function(BaseTool):
    """
    Template function to be called with argument names from `context`.

    Methods
    -------
    substitute
        Call a function.
    """

    @staticmethod
    def get_func_arg_names(
        func: tp.Callable,
        arg_kind: tp.Optional[int | tp.Tuple[int, ...]] = None,
        req_only: tp.Optional[bool] = False,
        opt_only: tp.Optional[bool] = False,
    ) -> tp.List[str]:
        """
        Get argument names of a function.

        Parameters
        ----------
        func : Callable
            The function whose argument names are to be retrieved.
        arg_kind : Optional[MaybeTuple[int]], optional
            Filter the kinds of arguments to retrieve:
            - If None, retrieves all arguments except *args and **kwargs.
            - If an int or tuple of ints, retrieves arguments of specific kinds:
              0 (POSITIONAL_ONLY), 1 (POSITIONAL_OR_KEYWORD), 2 (VAR_POSITIONAL),
              3 (KEYWORD_ONLY), and 4 (VAR_KEYWORD).
        req_only : bool, optional
            If True, retrieve only required (non-default) arguments.
        opt_only : bool, optional
            If True, retrieve only optional (defaulted) arguments.

        Returns
        -------
        List[str]
            A list of argument names based on the specified filters.

        Notes
        -----
        This function inspects the signature of the provided function `func` and
        filters its parameters based on the specified criteria. The `arg_kind`
        parameter can be used to filter arguments by their kind 
        (e.g., POSITIONAL_ONLY, KEYWORD_ONLY, etc.). The `req_only` and `opt_only` 
        parameters control whether to retrieve only required or optional arguments, 
        respectively.

        """
        signature = inspect.signature(func)
        if arg_kind is not None and isinstance(arg_kind, int):
            arg_kind = (arg_kind,)
        arg_names = []
        for p in signature.parameters.values():
            if arg_kind is None:
                if p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD:
                    continue
            else:
                if p.kind not in arg_kind:
                    continue
            if req_only and p.default is not inspect.Parameter.empty:
                continue
            if opt_only and p.default is inspect.Parameter.empty:
                continue
            arg_names.append(p.name)
        return arg_names

    def substitute(
        self,
        context: tp.Optional[tp.Dict[str, tp.Any]] = None,
        strict: tp.Optional[bool] = None,
        eval_id: int = 0,
    ) -> tp.Any:
        """
        Call `RepFunc.template` as a function.

        Parameters
        ----------
        context : tp.Dict[str, tp.Any], optional
            Additional context mapping to merge, by default None.
        eval_id : int, optional
            Evaluation identifier to append to the context, by default 0.

        Returns
        -------
        Any
            Result of the function call.
        """
        if not self.meets_eval_id(eval_id):
            return self

        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        func_arg_names = self.get_func_arg_names(self.template)
        func_kwargs = dict()
        for k, v in context.items():
            if k in func_arg_names:
                func_kwargs[k] = v

        try:
            return self.template(**func_kwargs)
        except TypeError as e:
            if strict:
                raise e
        return self


def substitute(
    obj: tp.Any,
    context: tp.Dict[str, tp.Any] = None,
    strict: tp.Optional[bool] = None,
    eval_id: tp.Optional[tp.Hashable] = None,
    **kwargs,
) -> tp.Any:
    """
    Traverses the object recursively and, if any template found, substitutes 
    it using a context.

    Parameters
    ----------
    obj : Any
        The object to be traversed and substituted.
    context : tp.Dict[str, tp.Any], optional
        The context to use for substitution (default is None).
    eval_id : Hashable, optional
        Evaluation identifier (default is None).

    Returns
    -------
    Any
        The substituted object.

    """

    def _match_func(k, v):
        return isinstance(v, (BaseTool, Template))

    def _replace_func(k, v):
        if isinstance(v, BaseTool):
            return v.substitute(context=context, strict=strict, eval_id=eval_id)
        if isinstance(v, Template):
            return v.substitute(context=context)

    return _find_and_replace_in_obj(obj, _match_func, _replace_func, **kwargs)


def _find_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    excl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    _key: tp.Optional[tp.Hashable] = None,
    _depth: int = 0,
    **kwargs,
) -> dict:
    """
    Find matches in an object in a recursive manner.

    Traverses dicts, tuples, lists and (frozen-)sets. Does not look for 
    matches in keys.

    Parameters
    ----------
    obj : Any
        The object to be searched.
    match_func : Callable
        The function to determine if a match is found.
    _key : Hashable, optional
        The current key in the traversal (default is None).
    _depth : int, optional
        The current recursion depth (default is 0).

    Returns
    -------
    dict
        A map of keys (represented by tuples) to their respective 
        values.

    """
    search_cfg = dict(
        excl_types=(list, set, frozenset),
        incl_types=None,
        max_len=None,
        max_depth=None,
    )
    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if match_func(_key, obj, **kwargs):
        return {_key: obj}

    if max_depth is None or _depth < max_depth:
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                return {}
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                match_dct = {}
                for k, v in obj.items():
                    new_key = k if _key is None else (
                        *_key, k) if isinstance(_key, tuple) else (_key, k)
                    match_dct.update(
                        _find_in_obj(
                            v,
                            match_func,
                            excl_types=excl_types,
                            incl_types=incl_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            _key=new_key,
                            _depth=_depth + 1,
                            **kwargs,
                        )
                    )
                return match_dct
        if isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                match_dct = {}
                for i, o in enumerate(obj):
                    new_key = i if _key is None else (
                        *_key, i) if isinstance(_key, tuple) else (_key, i)
                    match_dct.update(
                        _find_in_obj(
                            o,
                            match_func,
                            excl_types=excl_types,
                            incl_types=incl_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            _key=new_key,
                            _depth=_depth + 1,
                            **kwargs,
                        )
                    )
                return match_dct
    return {}


def _any_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    excl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    _key: tp.Optional[tp.Hashable] = None,
    _depth: int = 0,
    **kwargs,
) -> bool:
    """
    Return whether there is any match in an object in a recursive 
    manner.

    Parameters
    ----------
    obj : Any
        The object to be searched.
    match_func : Callable
        The function to determine if a match is found.
    _key : Hashable, optional
        The current key in the traversal (default is None).
    _depth : int, optional
        The current recursion depth (default is 0).

    Returns
    -------
    bool
        True if any match is found, otherwise False.

    """
    search_cfg = dict(
        excl_types=(list, set, frozenset),
        incl_types=None,
        max_len=None,
        max_depth=None,
    )

    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if match_func(_key, obj, **kwargs):
        return True
    if max_depth is None or _depth < max_depth:
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                return False
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                for k, v in obj.items():
                    new_key = (
                        k
                        if _key is None
                        else (*_key, k)
                        if isinstance(_key, tuple)
                        else (_key, k)
                    )
                    if _find_in_obj(
                        v,
                        match_func,
                        excl_types=excl_types,
                        incl_types=incl_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        _key=new_key,
                        _depth=_depth + 1,
                        **kwargs,
                    ):
                        return True
        if isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                for i, o in enumerate(obj):
                    new_key = i if _key is None else (
                        *_key, i) if isinstance(_key, tuple) else (_key, i)
                    if _find_in_obj(
                        o,
                        match_func,
                        excl_types=excl_types,
                        incl_types=incl_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        _key=new_key,
                        _depth=_depth + 1,
                        **kwargs,
                    ):
                        return True
    return False


def _find_and_replace_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    replace_func: tp.Callable,
    excl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    make_copy: bool = True,
    check_any_first: bool = True,
    _key: tp.Optional[tp.Hashable] = None,
    _depth: int = 0,
    **kwargs,
) -> tp.Any:
    """
    Find and replace matches in an object in a recursive manner.

    Parameters
    ----------
    obj : Any
        The object to be searched and modified.
    match_func : Callable
        The function to determine if a match is found.
    replace_func : Callable
        The function to perform replacement of matches.
    make_copy : bool, optional
        Whether to create a copy of the object (default is True).
    check_any_first : bool, optional
        Whether to check for any matches before replacing (default is True).
    _key : Hashable, optional
        The current key in the traversal (default is None).
    _depth : int, optional
        The current recursion depth (default is 0).

    Returns
    -------
    Any
        The modified object after replacement.

    Notes
    -----
    If the object is deep (such as a dict or a list), creates a copy of it 
    if any match found inside, thus losing the reference to the original.
    Make sure to do a deep or hybrid copy of the object before proceeding 
    for consistent behavior, or disable `make_copy` to override the 
    original in place.

    """
    search_cfg = dict(
        excl_types=(list, set, frozenset),
        incl_types=None,
        max_len=None,
        max_depth=None,
    )

    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if check_any_first and not _any_in_obj(
        obj,
        match_func,
        excl_types=excl_types,
        incl_types=incl_types,
        max_len=max_len,
        max_depth=max_depth,
        _key=_key,
        _depth=_depth,
        **kwargs,
    ):
        return obj

    if match_func(_key, obj, **kwargs):
        return replace_func(_key, obj, **kwargs)
    if max_depth is None or _depth < max_depth:
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                return obj
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for k, v in obj.items():
                    new_key = k if _key is None else (
                        *_key, k) if isinstance(_key, tuple) else (_key, k)
                    obj[k] = _find_and_replace_in_obj(
                        v,
                        match_func,
                        replace_func,
                        excl_types=excl_types,
                        incl_types=incl_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        make_copy=make_copy,
                        check_any_first=False,
                        _key=new_key,
                        _depth=_depth + 1,
                        **kwargs,
                    )
                return obj
        if isinstance(obj, list):
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for i in range(len(obj)):
                    new_key = i if _key is None else (
                        *_key, i) if isinstance(_key, tuple) else (_key, i)
                    obj[i] = _find_and_replace_in_obj(
                        obj[i],
                        match_func,
                        replace_func,
                        excl_types=excl_types,
                        incl_types=incl_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        make_copy=make_copy,
                        check_any_first=False,
                        _key=new_key,
                        _depth=_depth + 1,
                        **kwargs,
                    )
                return obj
        if isinstance(obj, (tuple, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                result = []
                for i, o in enumerate(obj):
                    new_key = i if _key is None else (
                        *_key, i) if isinstance(_key, tuple) else (_key, i)
                    result.append(
                        _find_and_replace_in_obj(
                            o,
                            match_func,
                            replace_func,
                            excl_types=excl_types,
                            incl_types=incl_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            make_copy=make_copy,
                            check_any_first=False,
                            _key=new_key,
                            _depth=_depth + 1,
                            **kwargs,
                        )
                    )
                if checks.is_namedtuple(obj):
                    return type(obj)(*result)
                return type(obj)(result)
    return obj
