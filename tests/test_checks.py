from collections import namedtuple
import numpy as np
import pandas as pd

from fold.utils import checks


def test_is_np_array():
    assert not checks.is_np_array(0)
    assert checks.is_np_array(np.array([0]))
    assert not checks.is_np_array(pd.Series([1, 2, 3]))
    assert not checks.is_np_array(pd.DataFrame([1, 2, 3]))


def test_is_pandas():
    assert not checks.is_pandas(0)
    assert not checks.is_pandas(np.array([0]))
    assert checks.is_pandas(pd.Series([1, 2, 3]))
    assert checks.is_pandas(pd.DataFrame([1, 2, 3]))


def test_is_series():
    assert not checks.is_series(0)
    assert not checks.is_series(np.array([0]))
    assert checks.is_series(pd.Series([1, 2, 3]))
    assert not checks.is_series(pd.DataFrame([1, 2, 3]))


def test_is_frame():
    assert not checks.is_frame(0)
    assert not checks.is_frame(np.array([0]))
    assert not checks.is_frame(pd.Series([1, 2, 3]))
    assert checks.is_frame(pd.DataFrame([1, 2, 3]))


def test_is_array():
    assert not checks.is_any_array(0)
    assert checks.is_any_array(np.array([0]))
    assert checks.is_any_array(pd.Series([1, 2, 3]))
    assert checks.is_any_array(pd.DataFrame([1, 2, 3]))


def test_is_sequence():
    assert checks.is_sequence([1, 2, 3])
    assert checks.is_sequence("123")
    assert not checks.is_sequence(0)
    assert not checks.is_sequence(dict(a=2).items())


def test_is_iterable():
    assert checks.is_iterable([1, 2, 3])
    assert checks.is_iterable("123")
    assert not checks.is_iterable(0)
    assert checks.is_iterable(dict(a=2).items())


def test_is_hashable():
    assert checks.is_hashable(2)
    assert not checks.is_hashable(np.asarray(2))


def test_is_index_equal():
    assert checks.is_index_equal(pd.Index([0]), pd.Index([0]))
    assert not checks.is_index_equal(pd.Index([0]), pd.Index([1]))
    assert not checks.is_index_equal(
        pd.Index([0], name="name"), pd.Index([0]))
    assert checks.is_index_equal(
        pd.Index([0], name="name"), pd.Index([0]), check_names=False)
    assert not checks.is_index_equal(
        pd.MultiIndex.from_arrays([[0], [1]]), pd.Index([0]))
    assert checks.is_index_equal(pd.MultiIndex.from_arrays(
        [[0], [1]]), pd.MultiIndex.from_arrays([[0], [1]]))
    assert checks.is_index_equal(
        pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
        pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
    )
    assert not checks.is_index_equal(
        pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
        pd.MultiIndex.from_arrays([[0], [1]], names=["name3", "name4"]),
    )


def test_is_default_index():
    assert checks.is_default_index(pd.DataFrame([[1, 2, 3]]).columns)
    assert checks.is_default_index(pd.Series([1, 2, 3]).to_frame().columns)
    assert checks.is_default_index(pd.Index([0, 1, 2]))
    assert not checks.is_default_index(pd.Index([0, 1, 2], name="name"))


def test_is_namedtuple():
    assert checks.is_namedtuple(namedtuple("Hello", ["world"])(*range(1)))
    assert not checks.is_namedtuple((0,))


def test_is_instance_of():
    class _A:
        pass

    class A:
        pass

    class B:
        pass

    class C(B):
        pass

    class D(A, C):
        pass

    d = D()

    assert not checks.is_instance_of(d, _A)
    assert checks.is_instance_of(d, A)
    assert checks.is_instance_of(d, B)
    assert checks.is_instance_of(d, C)
    assert checks.is_instance_of(d, D)
    assert checks.is_instance_of(d, object)

    assert not checks.is_instance_of(d, "_A")
    assert checks.is_instance_of(d, "A")
    assert checks.is_instance_of(d, "B")
    assert checks.is_instance_of(d, "C")
    assert checks.is_instance_of(d, "D")
    assert checks.is_instance_of(d, "object")


def test_is_subclass_of():
    class _A:
        pass

    class A:
        pass

    class B:
        pass

    class C(B):
        pass

    class D(A, C):
        pass

    assert not checks.is_subclass_of(D, _A)
    assert checks.is_subclass_of(D, A)
    assert checks.is_subclass_of(D, B)
    assert checks.is_subclass_of(D, C)
    assert checks.is_subclass_of(D, D)
    assert checks.is_subclass_of(D, object)

    assert not checks.is_subclass_of(D, "_A")
    assert checks.is_subclass_of(D, "A")
    assert checks.is_subclass_of(D, "B")
    assert checks.is_subclass_of(D, "C")
    assert checks.is_subclass_of(D, "D")
    assert checks.is_subclass_of(D, "object")
