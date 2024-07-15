from functools import partial

import numpy as np
import pandas as pd
import pytest

from fold import *

seed = 42

index = pd.date_range("2020-01-01", "2020-02-01", inclusive="left")

assert_index_equal = partial(
    pd.testing.assert_index_equal,
    rtol=1e-06,
    atol=0
)

assert_series_equal = partial(
    pd.testing.assert_series_equal,
    rtol=1e-06,
    atol=0
)

assert_frame_equal = partial(
    pd.testing.assert_frame_equal,
    rtol=1e-06,
    atol=0
)


def test_base_model():
    np.testing.assert_array_equal(
        BaseModel(index, [0.5]).splits_arr,
        np.array([[slice(0, 15, None), slice(15, 31, None)]], dtype=object),
    )
    assert_index_equal(
        BaseModel(index, [0.5]).split_labels,
        pd.RangeIndex(start=0, stop=1, step=1, name="split")
    )
    assert_index_equal(
        BaseModel(index, [0.5]).sample_labels,
        pd.Index(["sample_0", "sample_1"], dtype="object", name="sample"),
    )
    assert BaseModel(index, [0.5]).ndim == 2
    np.testing.assert_array_equal(
        BaseModel(index, [[0.5]]).splits_arr,
        np.array([[slice(0, 15, None)]], dtype=object)
    )
    assert_index_equal(
        BaseModel(index, [[0.5]]).split_labels,
        pd.RangeIndex(start=0, stop=1, step=1, name="split"),
    )
    assert_index_equal(
        BaseModel(index, [[0.5]]).sample_labels,
        pd.Index(["sample_0"], dtype="object", name="sample"),
    )
    assert BaseModel(index, [[0.5]]).ndim == 2
    np.testing.assert_array_equal(
        BaseModel(index, [[0.5]], fix_ranges=False).splits_arr,
        np.array([[RelativePeriod(length=0.5)]], dtype=object),
    )
    assert BaseModel(
        index, [[0.5]],
        range_format="mask"
    ).splits_arr.shape == (1, 1)
    np.testing.assert_array_equal(
        BaseModel(
            index, [[0.5]],
            range_format="mask"
        )
        .splits_arr[0, 0]
        .period,
        np.array([*[True] * 15, *[False] * 16]),
    )
    np.testing.assert_array_equal(
        BaseModel(index, [[0.5], [1.0]]).splits_arr,
        np.array([[slice(0, 15, None)], [slice(0, 31, None)]], dtype=object),
    )
    np.testing.assert_array_equal(
        BaseModel(index, [[0.25, 0.5], [0.75, 1.0]]).splits_arr,
        np.array(
            [
                [slice(0, 7, None), slice(7, 19, None)],
                [slice(0, 23, None), slice(23, 31, None)],
            ],
            dtype=object,
        ),
    )
    assert_index_equal(
        BaseModel(
            index,
            [[0.25, 0.5], [0.75, 1.0]],
            split_labels=["s1", "s2"]
        ).split_labels,
        pd.Index(["s1", "s2"], name="split"),
    )
    assert_index_equal(
        BaseModel(
            index,
            [[0.25, 0.5], [0.75, 1.0]],
            split_labels=pd.Index(["s1", "s2"], name="my_split"),
        ).split_labels,
        pd.Index(["s1", "s2"], name="my_split"),
    )
    assert_index_equal(
        BaseModel(
            index, [[0.25, 0.5], [0.75, 1.0]],
            sample_labels=["s1", "s2"]
        ).sample_labels,
        pd.Index(["s1", "s2"], name="sample"),
    )
    assert_index_equal(
        BaseModel(
            index,
            [[0.25, 0.5], [0.75, 1.0]],
            sample_labels=pd.Index(["s1", "s2"], name="my_set"),
        ).sample_labels,
        pd.Index(["s1", "s2"], name="my_set"),
    )

def test_function_split():
    def split_func(split_idx, x, y=15):
        if split_idx == 0:
            return slice(x, y)
        return None

    np.testing.assert_array_equal(
        FunctionSplit(
            index,
            split_func,
            split_args=(Key("split_idx"), 10),
            split_kwargs=dict(y=20),
        ).splits_arr,
        np.array([[slice(10, 20, None)]], dtype=object),
    )

    def split_func(split_idx, splits, bounds):
        if split_idx == 0:
            return [slice(0, 5), slice(5, 10)]
        if split_idx == 1:
            return slice(splits[-1][-1].stop, 15), slice(15, 20)
        if split_idx == 2:
            return slice(bounds[-1][-1][1], 25), slice(25, 30)
        return None

    np.testing.assert_array_equal(
        FunctionSplit(
            index,
            split_func,
            split_args=(Key("split_idx"), Key("splits"), Key("bounds")),
        ).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(10, 15, None), slice(15, 20, None)],
                [slice(20, 25, None), slice(25, 30, None)],
            ],
            dtype=object,
        ),
    )

    def split_func(split_idx, splits, bounds):
        if split_idx == 0:
            return [slice(0, 10)]
        if split_idx == 1:
            return slice(splits[-1][-1].stop, 20)
        if split_idx == 2:
            return slice(bounds[-1][-1][1], 30)
        return None

    np.testing.assert_array_equal(
        FunctionSplit(
            index,
            split_func,
            split_args=(Key("split_idx"), Key("splits"), Key("bounds")),
            split=0.5,
        ).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(10, 15, None), slice(15, 20, None)],
                [slice(20, 25, None), slice(25, 30, None)],
            ],
            dtype=object,
        ),
    )

    def split_func(split_idx):
        if split_idx == 0:
            return 0.5
        if split_idx == 1:
            return 1.0
        return None

    np.testing.assert_array_equal(
        FunctionSplit(
            index,
            Key("split_func", context=dict(split_func=split_func)),
            split_args=(Key("split_idx"),),
            fix_ranges=False,
        ).splits_arr,
        np.array(
            [[RelativePeriod(length=0.5)],
             [RelativePeriod(length=1.0)]]
        ),
    )

    def split_func(split_idx):
        if split_idx == 0:
            return 0.5
        if split_idx == 1:
            return 0.5, 1.0
        return None

    with pytest.raises(Exception):
        FunctionSplit(
            index,
            split_func,
            split_args=(Key("split_idx"),),
        )

def test_single_split():
    np.testing.assert_array_equal(
        SingleSplit(index, 0.5).splits_arr,
        np.array([[slice(0, 15, None), slice(15, 31, None)]], dtype=object),
    )

def test_rolling_split():
    with pytest.raises(Exception):
        RollingSplit(index, -1)
    with pytest.raises(Exception):
        RollingSplit(index, 0)
    with pytest.raises(Exception):
        RollingSplit(index, 1.5)
    np.testing.assert_array_equal(
        RollingSplit(index, 0.5).splits_arr,
        np.array([[slice(0, 15, None)], [slice(15, 30, None)]], dtype=object),
    )
    np.testing.assert_array_equal(
        RollingSplit(index, 0.7).splits_arr,
        np.array([[slice(0, 21, None)]], dtype=object),
    )
    np.testing.assert_array_equal(
        RollingSplit(index, 1.0).splits_arr,
        np.array([[slice(0, 31, None)]], dtype=object),
    )
    np.testing.assert_array_equal(
        RollingSplit(index, 10).splits_arr,
        np.array([[slice(0, 10, None)], [slice(10, 20, None)],
                 [slice(20, 30, None)]], dtype=object),
    )
    with pytest.raises(Exception):
        RollingSplit(index, 10, offset=1.5)
    np.testing.assert_array_equal(
        RollingSplit(index, 10, offset=0.1).splits_arr,
        np.array([[slice(0, 10, None)], [slice(11, 21, None)]], dtype=object),
    )
    np.testing.assert_array_equal(
        RollingSplit(index, 10, offset=-1).splits_arr,
        np.array([[slice(0, 10, None)], [slice(9, 19, None)],
                 [slice(18, 28, None)]], dtype=object),
    )
    np.testing.assert_array_equal(
        RollingSplit(index, 10, offset=-0.1).splits_arr,
        np.array([[slice(0, 10, None)], [slice(9, 19, None)],
                 [slice(18, 28, None)]], dtype=object),
    )
    np.testing.assert_array_equal(
        RollingSplit(index, 10, offset=10,
                     offset_anchor="prev_start").splits_arr,
        np.array([[slice(0, 10, None)], [slice(10, 20, None)],
                 [slice(20, 30, None)]], dtype=object),
    )
    with pytest.raises(Exception):
        RollingSplit(index, 10, offset_anchor="prev_start")
    np.testing.assert_array_equal(
        RollingSplit(index, 10, split=0.5).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(5, 10, None), slice(10, 15, None)],
                [slice(10, 15, None), slice(15, 20, None)],
                [slice(15, 20, None), slice(20, 25, None)],
                [slice(20, 25, None), slice(25, 30, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RollingSplit(
            index,
            10,
            split=0.5,
            offset_anchor_set=-1
        ).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(10, 15, None), slice(15, 20, None)],
                [slice(20, 25, None), slice(25, 30, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RollingSplit(
            index,
            10,
            split=0.5,
            offset_anchor_set=None
        ).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(10, 15, None), slice(15, 20, None)],
                [slice(20, 25, None), slice(25, 30, None)],
            ],
            dtype=object,
        ),
    )

def test_rolling_split():
    with pytest.raises(Exception):
        RollingNumberSplit(index, 5, length=-1)
    with pytest.raises(Exception):
        RollingNumberSplit(index, 5, length=0)
    with pytest.raises(Exception):
        RollingNumberSplit(index, 5, length=1.5)
    np.testing.assert_array_equal(
        RollingNumberSplit(index, 5, 10).splits_arr,
        np.array(
            [
                [slice(0, 10, None)],
                [slice(5, 15, None)],
                [slice(10, 20, None)],
                [slice(16, 26, None)],
                [slice(21, 31, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RollingNumberSplit(index, 5, length=10, split=0.5).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(5, 10, None), slice(10, 15, None)],
                [slice(10, 15, None), slice(15, 20, None)],
                [slice(16, 21, None), slice(21, 26, None)],
                [slice(21, 26, None), slice(26, 31, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RollingNumberSplit(index, 40, length=10).splits_arr,
        np.array([*[[slice(i, i + 10)] for i in range(22)]], dtype=object),
    )
    np.testing.assert_array_equal(
        RollingNumberSplit(index, 5, length="5 days").splits_arr,
        RollingNumberSplit(index, 5, length=5).splits_arr,
    )

def test_expanding_split():
    with pytest.raises(Exception):
        ExpandingSplit(index, -1, 1)
    with pytest.raises(Exception):
        ExpandingSplit(index, 0, 1)
    with pytest.raises(Exception):
        ExpandingSplit(index, 1.5, 1)
    with pytest.raises(Exception):
        ExpandingSplit(index, 1, -1)
    with pytest.raises(Exception):
        ExpandingSplit(index, 1, 0)
    with pytest.raises(Exception):
        ExpandingSplit(index, 1, 1.5)
    np.testing.assert_array_equal(
        ExpandingSplit(index, 10, 5).splits_arr,
        np.array(
            [
                [slice(0, 10, None)],
                [slice(0, 15, None)],
                [slice(0, 20, None)],
                [slice(0, 25, None)],
                [slice(0, 30, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        ExpandingSplit(index, 0.5, 5).splits_arr,
        np.array(
            [
                [slice(0, 15, None)],
                [slice(0, 20, None)],
                [slice(0, 25, None)],
                [slice(0, 30, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        ExpandingSplit(index, 10, 1 / 6).splits_arr,
        np.array(
            [
                [slice(0, 10, None)],
                [slice(0, 15, None)],
                [slice(0, 20, None)],
                [slice(0, 25, None)],
                [slice(0, 30, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        ExpandingSplit(index, 10, 5, split=0.5).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(0, 7, None), slice(7, 15, None)],
                [slice(0, 10, None), slice(10, 20, None)],
                [slice(0, 12, None), slice(12, 25, None)],
                [slice(0, 15, None), slice(15, 30, None)],
            ],
            dtype=object,
        ),
    )

def test_expanding_number_split():
    with pytest.raises(Exception):
        ExpandingNumberSplit(index, 5, min_length=-1)
    with pytest.raises(Exception):
        ExpandingNumberSplit(index, 5, min_length=0)
    with pytest.raises(Exception):
        ExpandingNumberSplit(index, 5, min_length=1.5)
    np.testing.assert_array_equal(
        ExpandingNumberSplit(index, 5).splits_arr,
        np.array(
            [
                [slice(0, 6, None)],
                [slice(0, 12, None)],
                [slice(0, 18, None)],
                [slice(0, 25, None)],
                [slice(0, 31, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        ExpandingNumberSplit(index, 5, min_length=10).splits_arr,
        np.array(
            [
                [slice(0, 10, None)],
                [slice(0, 15, None)],
                [slice(0, 20, None)],
                [slice(0, 26, None)],
                [slice(0, 31, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        ExpandingNumberSplit(index, 5, min_length=10,
                             split=0.5).splits_arr,
        np.array(
            [
                [slice(0, 5, None), slice(5, 10, None)],
                [slice(0, 7, None), slice(7, 15, None)],
                [slice(0, 10, None), slice(10, 20, None)],
                [slice(0, 13, None), slice(13, 26, None)],
                [slice(0, 15, None), slice(15, 31, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        ExpandingNumberSplit(index, 40, min_length=10).splits_arr,
        np.array([*[[slice(0, i + 10)] for i in range(22)]], dtype=object),
    )
    np.testing.assert_array_equal(
        ExpandingNumberSplit(index, 5, min_length="10 days").splits_arr,
        ExpandingNumberSplit(index, 5, min_length=10).splits_arr,
    )

def test_random_number_split():
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, min_start=-1)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, min_start=-0.1)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, min_start=1.5)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, min_start=100)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_end=-1)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_end=0)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_end=1.5)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_end=100)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, -1)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 0)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 1.5)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 100)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_length=-1)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_length=0)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_length=1.5)
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 10, max_length=100)
    np.testing.assert_array_equal(
        RandomNumberSplit(index, 5, 5, 10, seed=seed).splits_arr,
        np.array(
            [
                [slice(20, 25, None)],
                [slice(10, 18, None)],
                [slice(21, 28, None)],
                [slice(18, 23, None)],
                [slice(2, 8, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(index, 5, 6, split=0.5, seed=seed).splits_arr,
        np.array(
            [
                [slice(2, 5, None), slice(5, 8, None)],
                [slice(20, 23, None), slice(23, 26, None)],
                [slice(17, 20, None), slice(20, 23, None)],
                [slice(11, 14, None), slice(14, 17, None)],
                [slice(11, 14, None), slice(14, 17, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(index, 5, 5, 10, min_start=20,
                          seed=seed).splits_arr,
        np.array(
            [
                [slice(25, 30, None)],
                [slice(21, 29, None)],
                [slice(24, 31, None)],
                [slice(24, 29, None)],
                [slice(20, 26, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(index, 5, 5, 10, min_start=10,
                          max_end=20, seed=seed).splits_arr,
        np.array(
            [
                [slice(14, 19, None)],
                [slice(11, 19, None)],
                [slice(13, 20, None)],
                [slice(14, 19, None)],
                [slice(10, 16, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index, 5, 5, 10, length_p_func=lambda i, x: np.arange(len(x)) / np.arange(len(x)).sum(), seed=seed
        ).splits_arr,
        np.array(
            [
                [slice(14, 24, None)],
                [slice(9, 19, None)],
                [slice(4, 14, None)],
                [slice(2, 12, None)],
                [slice(15, 25, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index, 5, 5, 10, start_p_func=lambda i, x: np.arange(len(x)) / np.arange(len(x)).sum(), seed=seed
        ).splits_arr,
        np.array(
            [
                [slice(18, 23, None)],
                [slice(21, 30, None)],
                [slice(8, 13, None)],
                [slice(22, 31, None)],
                [slice(20, 29, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            length_choice_func=lambda i, x: np.random.choice(
                x, p=np.arange(len(x)) / np.arange(len(x)).sum()),
            seed=seed,
        ).splits_arr,
        np.array(
            [
                [slice(2, 10, None)],
                [slice(17, 27, None)],
                [slice(14, 24, None)],
                [slice(10, 19, None)],
                [slice(10, 17, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            start_choice_func=lambda i, x: np.random.choice(
                x,
                p=np.arange(len(x)) / np.arange(len(x)).sum()
            ),
            seed=seed,
        ).splits_arr,
        np.array(
            [
                [slice(16, 21, None)],
                [slice(22, 31, None)],
                [slice(20, 28, None)],
                [slice(19, 26, None)],
                [slice(10, 17, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            min_start="2020-01-03",
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            min_start=2,
            seed=seed
        ).splits_arr,
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            max_end="2020-01-29",
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            max_end=28,
            seed=seed
        ).splits_arr,
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            "5 days",
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            5,
            seed=seed
        ).splits_arr,
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            "5 days",
            "10 days",
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            seed=seed
        ).splits_arr,
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            "5 days",
            "10 days",
            min_start="2020-01-03",
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            min_start=2,
            seed=seed
        ).splits_arr,
    )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            "5 days",
            "10 days",
            max_end="2020-01-29",
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            5,
            10,
            max_end=28,
            seed=seed
        ).splits_arr,
    )
    with pytest.raises(Exception):
        RandomNumberSplit(
            index,
            5,
            1,
            min_start=index[0] - pd.Timedelta(days=1),
            seed=seed
        )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            1,
            min_start=index[0],
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            1,
            min_start=0,
            seed=seed
        ).splits_arr,
    )
    with pytest.raises(Exception):
        RandomNumberSplit(
            index,
            5,
            1,
            min_start=index[-1] + pd.Timedelta(days=1),
            seed=seed
        )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            1,
            min_start=index[-1],
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            1,
            min_start=len(index) - 1,
            seed=seed
        ).splits_arr,
    )
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, 1, max_end=index[0], seed=seed)
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            1,
            max_end=index[0] + pd.Timedelta(days=1),
            seed=seed
        ).splits_arr,
        RandomNumberSplit(index, 5, 1, max_end=1, seed=seed).splits_arr,
    )
    with pytest.raises(Exception):
        RandomNumberSplit(
            index,
            5,
            1,
            max_end=index[-1] + pd.Timedelta(days=2),
            seed=seed
        )
    np.testing.assert_array_equal(
        RandomNumberSplit(
            index,
            5,
            1,
            max_end=index[-1] + pd.Timedelta(days=1),
            seed=seed
        ).splits_arr,
        RandomNumberSplit(
            index,
            5,
            1,
            max_end=len(index),
            seed=seed
        ).splits_arr,
    )
    with pytest.raises(Exception):
        RandomNumberSplit(
            index,
            5,
            "5 days",
            max_end="2020-01-05",
            seed=seed
        )
    with pytest.raises(Exception):
        RandomNumberSplit(index, 5, "5 days", "4 days", seed=seed)

def test_interval_split():
    np.testing.assert_array_equal(
        IntervalSplit(index, every="W").splits_arr,
        np.array(
            [
                [slice(5, 12, None)],
                [slice(12, 19, None)],
                [slice(19, 26, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        IntervalSplit(index, every="W", split=0.5).splits_arr,
        np.array(
            [
                [slice(5, 8, None), slice(8, 12, None)],
                [slice(12, 15, None), slice(15, 19, None)],
                [slice(19, 22, None), slice(22, 26, None)],
            ],
            dtype=object,
        ),
    )

def test_grouper_split():
    np.testing.assert_array_equal(
        GrouperSplit(index, by="W").splits_arr,
        np.array(
            [
                [slice(0, 6, None)],
                [slice(6, 13, None)],
                [slice(13, 20, None)],
                [slice(20, 27, None)],
                [slice(27, 31, None)],
            ],
            dtype=object,
        ),
    )
    assert_index_equal(
        GrouperSplit(index, by="W").split_labels,
        pd.PeriodIndex(
            [
                "2019-12-31/2020-01-06",
                "2020-01-07/2020-01-13",
                "2020-01-14/2020-01-20",
                "2020-01-21/2020-01-27",
                "2020-01-28/2020-02-02",
            ],
            dtype="period[W-MON]",
        ),
    )
    np.testing.assert_array_equal(
        GrouperSplit(index, by="W", split=0.5).splits_arr,
        np.array(
            [
                [slice(0, 3, None), slice(3, 6, None)],
                [slice(6, 9, None), slice(9, 13, None)],
                [slice(13, 16, None), slice(16, 20, None)],
                [slice(20, 23, None), slice(23, 27, None)],
                [slice(27, 29, None), slice(29, 31, None)],
            ],
            dtype=object,
        ),
    )

def test_scikit_learn():
    from sklearn.model_selection import TimeSeriesSplit
    sk_model = TimeSeriesSplit(n_splits=5)
    np.testing.assert_array_equal(
        SklearnFold(index, sk_model).splits_arr,
        np.array(
            [
                [slice(0, 6, None), slice(6, 11, None)],
                [slice(0, 11, None), slice(11, 16, None)],
                [slice(0, 16, None), slice(16, 21, None)],
                [slice(0, 21, None), slice(21, 26, None)],
                [slice(0, 26, None), slice(26, 31, None)],
            ],
            dtype=object,
        ),
    )

def test_purged_walkforward():
    np.testing.assert_array_equal(
        PurgedWalkForwardSplit(index, n_folds=5).splits_arr,
        np.array(
            [
                [slice(0, 13, None), slice(13, 19, None)],
                [slice(0, 19, None), slice(19, 25, None)],
                [slice(0, 25, None), slice(25, 31, None)],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        PurgedWalkForwardSplit(
            index,
            n_folds=5,
            pred_times=index,
            eval_times=index + pd.Timedelta(days=3),
        ).splits_arr,
        np.array(
            [
                [slice(0, 10, None), slice(13, 19, None)],
                [slice(0, 16, None), slice(19, 25, None)],
                [slice(0, 22, None), slice(25, 31, None)],
            ],
            dtype=object,
        ),
    )

def test_purged_kfold():
    splits_arr = PurgedKFold(index, n_folds=4).splits_arr
    assert splits_arr[0][0] == slice(0, 16, None)
    assert splits_arr[0][1] == slice(16, 31, None)
    np.testing.assert_array_equal(
        splits_arr[1][0].period,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[1][1].period,
        np.array(
            [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[2][0].period,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    assert splits_arr[2][1] == slice(8, 24, None)
    assert splits_arr[3][0] == slice(8, 24, None)
    np.testing.assert_array_equal(
        splits_arr[3][1].period,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[4][0].period,
        np.array(
            [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[4][1].period,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
        ),
    )
    assert splits_arr[5][0] == slice(16, 31, None)
    assert splits_arr[5][1] == slice(0, 16, None)

    splits_arr = PurgedKFold(
        index,
        n_folds=4,
        pred_times=index,
        eval_times=index + pd.Timedelta(days=3),
    ).splits_arr
    assert splits_arr[0][0] == slice(0, 13, None)
    assert splits_arr[0][1] == slice(16, 31, None)
    np.testing.assert_array_equal(
        splits_arr[1][0].period,
        np.array(
            [0, 1, 2, 3, 4, 19, 20]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[1][1].period,
        np.array(
            [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[2][0].period,
        np.array(
            [0, 1, 2, 3, 4, 27, 28, 29, 30]
        ),
    )
    assert splits_arr[2][1] == slice(8, 24, None)
    assert splits_arr[3][0] == slice(11, 21, None)
    np.testing.assert_array_equal(
        splits_arr[3][1].period,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[4][0].period,
        np.array(
            [11, 12, 27, 28, 29, 30]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[4][1].period,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
        ),
    )
    assert splits_arr[5][0] == slice(19, 31, None)
    assert splits_arr[5][1] == slice(0, 16, None)

    splits_arr = PurgedKFold(
        index,
        n_folds=4,
        embargo_td="2 days",
        pred_times=index,
        eval_times=index + pd.Timedelta(days=3),
    ).splits_arr
    assert splits_arr[0][0] == slice(0, 13, None)
    assert splits_arr[0][1] == slice(16, 31, None)
    assert splits_arr[1][0] == slice(0, 5, None)
    np.testing.assert_array_equal(
        splits_arr[1][1].period,
        np.array(
            [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    np.testing.assert_array_equal(
        splits_arr[2][0].period,
        np.array(
            [0,  1,  2,  3,  4, 29, 30]
        ),
    )
    assert splits_arr[2][1] == slice(8, 24, None)
    assert splits_arr[3][0] == slice(13, 21, None)
    np.testing.assert_array_equal(
        splits_arr[3][1].period,
        np.array(
            [0,  1,  2,  3,  4,  5,  6,  7, 24, 25, 26, 27, 28, 29, 30]
        ),
    )
    assert splits_arr[4][0] == slice(29, 31, None)
    np.testing.assert_array_equal(
        splits_arr[4][1].period,
        np.array(
            [0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 18, 19, 20, 21, 22, 23]
        ),
    )
    assert splits_arr[5][0] == slice(21, 31, None)
    assert splits_arr[5][1] == slice(0, 16, None)

def test_period_split():
    pass

def test_calendar_split():
    pass

def test_period_transformer():
    assert PeriodTransformer(
        Key("sample", context=dict(sample=slice(10, 20))),
        index=index
    ).period == slice(10, 20)
    assert PeriodTransformer(
        lambda index: slice(10, len(index)),
        index=index
    ).period == slice(10, len(index))
    with pytest.raises(Exception):
        PeriodTransformer(
            10,
            index=index
        )
    with pytest.raises(Exception):
        PeriodTransformer(
            0.5,
            index=index
        )
    with pytest.raises(Exception):
        PeriodTransformer(
            RelativePeriod(),
            index=index
        )
    assert PeriodTransformer(
        slice(10, len(index)),
        index=index
    ).period == slice(10, len(index))
    assert PeriodTransformer(
        slice(None),
        index=index
    ).period == slice(0, len(index))
    assert PeriodTransformer(
        slice(None, len(index)),
        index=index
    ).period == slice(0, len(index))
    assert PeriodTransformer(
        slice(0, len(index)),
        index=index
    ).period == slice(0, len(index))
    with pytest.raises(Exception):
        PeriodTransformer(
            slice(0, len(index), 2),
            index=index
        )
    assert PeriodTransformer(
        slice(-5, 0),
        index=index
    ).period == slice(len(index) - 5, len(index))
    with pytest.raises(Exception):
        PeriodTransformer(
            slice(-5, 1),
            index=index
        )
    with pytest.raises(Exception):
        PeriodTransformer(
            slice(0, 0),
            index=index
        ).period
    np.testing.assert_array_equal(
        PeriodTransformer(
            slice(5, 10),
            range_format="indices",
            index=index
        ).period,
        np.array([5, 6, 7, 8, 9]),
    )
    mask = np.full(len(index), False)
    mask[5:10] = True
    np.testing.assert_array_equal(
        PeriodTransformer(
            slice(5, 10),
            range_format="mask",
            index=index
        ).period,
        mask,
    )
    assert PeriodTransformer(
        slice(None, index[4]),
        index=index
    ).period == slice(0, 4)
    assert PeriodTransformer(
        slice(index[1], None),
        index=index
    ).period == slice(1, len(index))
    assert PeriodTransformer(
        slice(1, index[4]),
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        slice(index[1], 4),
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        slice(index[1], index[4]),
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        slice(index[0] - pd.Timedelta(days=1),
              index[-1] + pd.Timedelta(days=1)),
        index=index
    ).period == slice(0, len(index))
    assert PeriodTransformer(
        slice(index[1].to_datetime64(), index[4].to_datetime64()),
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        slice(str(index[1].to_datetime64()),
              str(index[4].to_datetime64())),
        index=index
    ).period == slice(1, 4)
    with pytest.raises(Exception):
        PeriodTransformer(
            slice("hello", str(index[4].to_datetime64())),
            index=index
        )
    with pytest.raises(Exception):
        PeriodTransformer(
            slice(str(index[1].to_datetime64()), "hello"),
            index=index
        )

    np.testing.assert_array_equal(
        PeriodTransformer(np.array([3, 2, 1]), index=index).period,
        np.array([3, 2, 1]),
    )
    assert PeriodTransformer(
        np.array([1, 2, 3]),
        index=index
    ).period == slice(1, 4)
    np.testing.assert_array_equal(
        PeriodTransformer(
            np.array([1, 2, 3]), range_format="indices",
            index=index
        ).period,
        np.array([1, 2, 3]),
    )
    mask = np.full(len(index), False)
    mask[[1, 2, 3]] = True
    np.testing.assert_array_equal(
        PeriodTransformer(
            np.array([1, 2, 3]),
            range_format="mask",
            index=index
        ).period,
        mask,
    )
    assert PeriodTransformer(
        np.array([1, 2, 3]),
        range_format="slice",
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        np.array([1, 2, 3]),
        range_format="slice_or_indices",
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        np.array([1, 2, 3]),
        range_format="slice_or_mask",
        index=index
    ).period == slice(1, 4)
    np.testing.assert_array_equal(
        PeriodTransformer(np.array([1, 3]), index=index).period,
        np.array([1, 3]),
    )
    mask = np.full(len(index), False)
    mask[[1, 3]] = True
    np.testing.assert_array_equal(
        PeriodTransformer(
            np.array([1, 3]),
            range_format="mask",
            index=index
        ).period,
        mask,
    )
    with pytest.raises(Exception):
        PeriodTransformer(
            np.array([1, 3]),
            range_format="slice",
            index=index
        )
    np.testing.assert_array_equal(
        PeriodTransformer(
            np.array([1, 3]),
            range_format="slice_or_indices",
            index=index
        ).period,
        np.array([1, 3]),
    )
    np.testing.assert_array_equal(
        PeriodTransformer(
            np.array([1, 3]),
            range_format="slice_or_mask",
            index=index
        ).period,
        mask,
    )
    assert PeriodTransformer(
        index[1:3],
        index=index
    ).period == slice(1, 3)
    assert PeriodTransformer(
        [index[1], index[2]],
        index=index
    ).period == slice(1, 3)
    assert PeriodTransformer(
        [str(index[1]), str(index[2])],
        index=index
    ).period == slice(1, 3)
    with pytest.raises(Exception):
        PeriodTransformer(["hello", "world"], index=index)

    mask = np.full(len(index), False)
    mask[[1, 2, 3]] = True
    assert PeriodTransformer(mask, index=index).period == slice(1, 4)
    np.testing.assert_array_equal(
        PeriodTransformer(
            mask,
            range_format="indices",
            index=index
        ).period,
        np.array([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        PeriodTransformer(
            mask,
            range_format="mask",
            index=index
        ).period,
        mask.astype(bool),
    )
    assert PeriodTransformer(
        mask,
        range_format="slice",
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        mask,
        range_format="slice_or_indices",
        index=index
    ).period == slice(1, 4)
    assert PeriodTransformer(
        mask,
        range_format="slice_or_mask",
        index=index
    ).period == slice(1, 4)

    mask = np.full(len(index), False)
    mask[[1, 3]] = True
    np.testing.assert_array_equal(
        PeriodTransformer(mask, index=index).period,
        mask.astype(bool),
    )
    np.testing.assert_array_equal(
        PeriodTransformer(
            mask,
            range_format="indices",
            index=index
        ).period,
        np.array([1, 3]),
    )
    with pytest.raises(Exception):
        PeriodTransformer(
            mask,
            range_format="slice",
            index=index
        )
    np.testing.assert_array_equal(
        PeriodTransformer(
            mask,
            range_format="slice_or_mask",
            index=index
        ).period,
        mask.astype(bool),
    )
    np.testing.assert_array_equal(
        PeriodTransformer(
            mask,
            range_format="slice_or_indices",
            index=index
        ).period,
        np.array([1, 3]),
    )

    with pytest.raises(Exception):
        PeriodTransformer(np.array([-1, -2]), index=index)

    with pytest.raises(Exception):
        PeriodTransformer(np.array([0, 100]), index=index)

    with pytest.raises(Exception):
        PeriodTransformer(np.array([100, 0]), index=index)

    with pytest.raises(Exception):
        PeriodTransformer(np.array([100, 200]), index=index)

    transformer = PeriodTransformer(
        FixedPeriod(
            Key(
                "sample",
                context=dict(sample=lambda index: slice(10, 20))
            )
        ).period,
        index=index,
    ).model_output
    assert transformer['period'] == slice(10, 20, None)
    assert transformer['range_format'] == None
    assert transformer['start'] == 10
    assert transformer['stop'] == 20
    assert transformer['length'] == 10

    mask = np.full(len(index), False)
    mask[[1, 2, 3]] = True
    transformer = PeriodTransformer(
        mask,
        index=index,
    ).model_output
    assert transformer['period'] == slice(1, 4, None)
    assert transformer['range_format'] == "slice_or_mask"
    assert transformer['start'] == 1
    assert transformer['stop'] == 4
    assert transformer['length'] == 3

    transformer = PeriodTransformer(
        np.array([1, 2, 3]),
        index=index,
    ).model_output
    assert transformer['period'] == slice(1, 4, None)
    assert transformer['range_format'] == "slice_or_indices"
    assert transformer['start'] == 1
    assert transformer['stop'] == 4
    assert transformer['length'] == 3

def test_split_period():
    with pytest.raises(Exception):
        SplitPeriod(20, index=index).split(slice(None))
        
    with pytest.raises(Exception):
        SplitPeriod(0.5, index=index).split(slice(None))
        
    with pytest.raises(Exception):
        SplitPeriod(RelativePeriod(), index=index).split(slice(None))
        
    assert SplitPeriod(
        slice(None),
        index=index
    ).split(slice(None)) == (
        slice(0, len(index)),
    )
    assert SplitPeriod(
        slice(None), 
        index=index
    ).split(0.75) == (
        slice(0, 23, None), 
        slice(23, 31, None)
    )
    assert SplitPeriod(
        slice(None),  
        index=index
    ).split(0.75, backwards=True) == (
        slice(0, 8, None),
        slice(8, 31, None),
    )
    assert SplitPeriod(
        slice(None), 
        index=index
    ).split(-0.25) == (
        slice(0, 24, None),
        slice(24, 31, None)
    )
    assert SplitPeriod(
        slice(None), 
        index=index
    ).split(-0.25, backwards=True)  == (
        slice(0, 7, None),
        slice(7, 31, None),
    )
    assert SplitPeriod(
        slice(None), 
        index=index
    ).split( 
        (
            RelativePeriod(
                length=10
            ), 
            RelativePeriod(
                length=5
            )
        )  
    ) == (
        slice(0, 10, None), 
        slice(10, 15, None)
    )
    assert SplitPeriod(
        slice(None), 
        index=index
    ).split( 
        ( 
            RelativePeriod(
                length=10
            ), 
            RelativePeriod(
                length=5
            )
        ),
        backwards=True, 
    ) == (
        slice(16, 26, None), 
        slice(26, 31, None)
    )
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split( 
        (
            RelativePeriod(
                length=10,
                offset_anchor="prev_start", 
                offset=10
            ),
            RelativePeriod(
                length=5, 
                offset_anchor="prev_start"
            ),
        ),            
    ) == (
        slice(10, 20, None), 
        slice(10, 15, None)
    )
    
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split(
        (
            RelativePeriod(
                length=10, 
                offset_anchor="prev_start", 
                offset=10
            ),
            RelativePeriod(
                length=5, 
                offset_anchor="prev_start"
            ),
        ),
        backwards=True, 
    ) == (
        slice(11, 21, None), 
        slice(26, 31, None)
    )
    
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split( 
        (
            np.arange(10, 20),
            RelativePeriod(
                length=5, 
                offset_anchor="prev_start"
            ),
        )
    ) == (
        slice(10, 20, None), 
        slice(10, 15, None)
    )
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split( 
        (
            RelativePeriod(
                length=10, 
                offset_anchor="prev_start",
                offset=10
            ),
            np.arange(26, 31),
        ),
        backwards=True,
    ) == (
        slice(11, 21, None), 
        slice(26, 31, None)
    )
    
    mask = np.full(len(index), False)
    mask[10:20] = True
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split( 
        (
            mask,
            RelativePeriod(
                length=5, 
                offset_anchor="prev_start"
            ),
        ),
    ) == (
        slice(10, 20, None), 
        slice(10, 15, None)
    )
    
    mask = np.full(len(index), False)
    mask[26:31] = True
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split(
        (
            RelativePeriod(
                length=10, 
                offset_anchor="prev_start", 
                offset=10
            ),
            mask,
        ),
        backwards=True,
    ) == (
        slice(11, 21, None), 
        slice(26, 31, None)
    )
    
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split( 
        (
            RelativePeriod(
                length=10, 
                offset_anchor="prev_start", 
                offset=10
            ),
            RelativePeriod(
                length=5, 
                is_gap=True
            ),
            RelativePeriod(
                length=5
            ),
        ),
    ) == (
        slice(10, 20, None), 
        slice(20, 25, None), # TO CHECK
        slice(25, 30, None) 
    )
    
    mask = np.full(len(index), False)
    mask[15:20] = True
    assert SplitPeriod(
        slice(None),
        index=index,
    ).split( 
        [
            slice(0, 5),
            slice(5, 10),
            np.arange(10, 15),
            mask,
        ],
    ) == (
        slice(0, 5, None),
        slice(5, 10, None),
        slice(10, 15, None),
        slice(15, 20, None),
    )
    target_mask = np.full((4, len(index)), False)
    target_mask[0, 0:5] = True
    target_mask[1, 5:10] = True
    target_mask[2, 10:15] = True
    target_mask[3, 15:20] = True
    np.testing.assert_array_equal(
        np.asarray(
            SplitPeriod(
                slice(None),
                index=index,
                range_format="mask",
            ).split(
                [
                    slice(0, 5),
                    slice(5, 10),
                    np.arange(10, 15),
                    mask,
                ],
            )
        ),
        target_mask,
    )
    
    mask = np.full(len(index), False)
    mask[[16, 18, 20]] = True
    new_split = SplitPeriod(
        slice(None),
        index=index,
    ).split( 
        [
            slice(0, 5),
            slice(5, 10),
            np.array([10, 12, 14]),
            mask,
        ],       
        backwards=True,
    )
    assert new_split[0] == slice(0, 5, None)
    assert new_split[1] == slice(5, 10, None)
    
    np.testing.assert_array_equal(new_split[2], np.array([10, 12, 14]))
    np.testing.assert_array_equal(new_split[3], np.array([16, 18, 20]))
    
    assert SplitPeriod(
        np.array([0, 2, 4, 5, 7, 8, 9, 11]), 
        index=index
    ).split(
        "by_gap", 
    ) == (
        slice(0, 1, None), 
        slice(2, 3, None), 
        slice(4, 6, None), 
        slice(7, 10, None), 
        slice(11, 12, None)
    )
    
    mask = np.full(len(index), False)
    mask[[0, 2, 4, 5, 7, 8, 9, 11]] = True
    assert SplitPeriod(
        mask, 
        index=index
    ).split(
        "by_gap"
    ) == (
        slice(0, 1, None), 
        slice(2, 3, None),
        slice(4, 6, None), 
        slice(7, 10, None), 
        slice(11, 12, None)
    )

def test_train_test_split():
    sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
    splitter = BaseModel(
        index,
        [
            [slice(0, 15), slice(10, 25)],
            [slice(5, 20), slice(15, None)],
        ],
    )
    new_obj = splitter.train_test_split(sr)
    assert_index_equal(
        new_obj.index,
        pd.MultiIndex.from_tuples(
            [
                (0, "sample_0"),
                (0, "sample_1"),
                (1, "sample_0"),
                (1, "sample_1")
            ],
            names=["split", "sample"]
        ),
    )

    assert_series_equal(
        new_obj.iloc[0],
        pd.Series(np.arange(5, 20), index=sr.index[5:20]),
    )
    assert_series_equal(
        new_obj.iloc[1],
        pd.Series(np.arange(15, 30), index=sr.index[15:30]),
    )
    assert_series_equal(
        new_obj.iloc[2],
        pd.Series(np.arange(10, 25), index=sr.index[10:25]),
    )
    assert_series_equal(
        new_obj.iloc[3],
        pd.Series(np.arange(20, 31), index=sr.index[20:31]),
    )

def test_get_bounds_arr():
    splitter = BaseModel(
        index,
        [
            [slice(0, 5), slice(5, 10)],
            [slice(10, 15), slice(15, 20)],
            [slice(20, 25), slice(25, None)],
        ],
    )
    np.testing.assert_array_equal(
        splitter.get_bounds_arr(),
        np.array(
            [[[0, 5], [5, 10]], [[10, 15], [15, 20]], [[20, 25], [25, 31]]]
        )
    )
    np.testing.assert_array_equal(
        splitter.get_bounds_arr(index_bounds=True),
        np.array(
            [
                [[1577836800000000000, 1578268800000000000], [
                    1578268800000000000, 1578700800000000000]],
                [[1578700800000000000, 1579132800000000000], [
                    1579132800000000000, 1579564800000000000]],
                [[1579564800000000000, 1579996800000000000], [
                    1579996800000000000, 1580515200000000000]],
            ],
            dtype="datetime64[ns]",
        ),
    )

def test_get_bounds():
    splitter = BaseModel(
        index,
        [
            [slice(0, 5), slice(5, 10)],
            [slice(10, 15), slice(15, 20)],
            [slice(20, 25), slice(25, None)],
        ],
    )
    assert_frame_equal(
        splitter.get_bounds(),
        pd.DataFrame(
            [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 31]],
            index=pd.MultiIndex.from_tuples(
                [
                    (0, "sample_0"), (0, "sample_1"), (1, "sample_0"),
                    (1, "sample_1"), (2, "sample_0"), (2, "sample_1")
                ],
                names=["split", "sample"],
            ),
            columns=pd.Index(["start", "end"],
                             dtype="object", name="bound"),
        ),
    )
    assert_frame_equal(
        splitter.index_bounds,
        pd.DataFrame(
            [
                [1577836800000000000, 1578268800000000000],
                [1578268800000000000, 1578700800000000000],
                [1578700800000000000, 1579132800000000000],
                [1579132800000000000, 1579564800000000000],
                [1579564800000000000, 1579996800000000000],
                [1579996800000000000, 1580515200000000000],
            ],
            dtype="datetime64[ns]",
            index=pd.MultiIndex.from_tuples(
                [
                    (0, "sample_0"), (0, "sample_1"), (1, "sample_0"),
                    (1, "sample_1"), (2, "sample_0"), (2, "sample_1")
                ],
                names=["split", "sample"],
            ),
            columns=pd.Index(["start", "end"],
                             dtype="object", name="bound"),
        ),
    )

def test_get_duration():
    splitter = BaseModel(
        index,
        [
            [slice(0, 5), slice(5, 10)],
            [slice(10, 15), slice(15, 20)],
            [slice(20, 25), slice(25, None)],
        ],
    )
    assert_series_equal(
        splitter.duration,
        pd.Series(
            [5, 5, 5, 5, 5, 6],
            index=pd.MultiIndex.from_tuples(
                [(0, "sample_0"), (0, "sample_1"), (1, "sample_0"),
                 (1, "sample_1"), (2, "sample_0"), (2, "sample_1")],
                names=["split", "sample"],
            ),
            name="duration",
        ),
    )
    assert_series_equal(
        splitter.index_duration,
        pd.Series(
            [
                432000000000000,
                432000000000000,
                432000000000000,
                432000000000000,
                432000000000000,
                518400000000000,
            ],
            dtype="timedelta64[ns]",
            index=pd.MultiIndex.from_tuples(
                [(0, "sample_0"), (0, "sample_1"), (1, "sample_0"),
                 (1, "sample_1"), (2, "sample_0"), (2, "sample_1")],
                names=["split", "sample"],
            ),
            name="duration",
        ),
    )

def test_get_iter_split_masks():
    splitter = BaseModel(
        index,
        [
            [slice(0, 5), slice(5, 10)],
            [slice(10, 15), slice(15, 20)],
            [slice(20, 25), slice(25, None)],
        ],
    )
    assert_frame_equal(
        list(splitter.get_iter_split_masks())[0],
        pd.DataFrame(
            [
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [False, True],
                [False, True],
                [False, True],
                [False, True],
                [False, True],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
            ],
            index=index,
            columns=pd.Index(["sample_0", "sample_1"],
                             dtype="object", name="sample"),
        ),
    )
    assert_frame_equal(
        list(splitter.get_iter_split_masks())[1],
        pd.DataFrame(
            [
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [False, True],
                [False, True],
                [False, True],
                [False, True],
                [False, True],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
            ],
            index=index,
            columns=pd.Index(["sample_0", "sample_1"],
                             dtype="object", name="sample"),
        ),
    )
    assert_frame_equal(
        list(splitter.get_iter_split_masks())[2],
        pd.DataFrame(
            [
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [False, True],
                [False, True],
                [False, True],
                [False, True],
                [False, True],
                [False, True],
            ],
            index=index,
            columns=pd.Index(["sample_0", "sample_1"],
                             dtype="object", name="sample"),
        ),
    )

def test_get_iter_set_masks():
    splitter = BaseModel(
        index,
        [
            [slice(0, 5), slice(5, 10)],
            [slice(10, 15), slice(15, 20)],
            [slice(20, 25), slice(25, None)],
        ],
    )
    assert_frame_equal(
        list(splitter.get_iter_sample_masks())[0],
        pd.DataFrame(
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
            index=index,
            columns=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
        ),
    )
    assert_frame_equal(
        list(splitter.get_iter_sample_masks())[1],
        pd.DataFrame(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
            ],
            index=index,
            columns=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
        ),
    )

def test_get_mask_arr():
    splitter = BaseModel(
        index,
        [
            [slice(0, 5), slice(5, 10)],
            [slice(10, 15), slice(15, 20)],
            [slice(20, 25), slice(25, None)],
        ],
    )
    np.testing.assert_array_equal(
        splitter.get_mask_arr(),
        np.array(
            [
                [
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                ],
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                ],
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                ],
            ]
        ),
    )

def test_get_mask():
    splitter = BaseModel(
        index,
        [
            [slice(0, 5), slice(5, 10)],
            [slice(10, 15), slice(15, 20)],
            [slice(20, 25), slice(25, None)],
        ],
    )
    assert_frame_equal(
        splitter.get_mask(),
        pd.DataFrame(
            [
                [True, False, False, False, False, False],
                [True, False, False, False, False, False],
                [True, False, False, False, False, False],
                [True, False, False, False, False, False],
                [True, False, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, False, True, False, False, False],
                [False, False, True, False, False, False],
                [False, False, True, False, False, False],
                [False, False, True, False, False, False],
                [False, False, True, False, False, False],
                [False, False, False, True, False, False],
                [False, False, False, True, False, False],
                [False, False, False, True, False, False],
                [False, False, False, True, False, False],
                [False, False, False, True, False, False],
                [False, False, False, False, True, False],
                [False, False, False, False, True, False],
                [False, False, False, False, True, False],
                [False, False, False, False, True, False],
                [False, False, False, False, True, False],
                [False, False, False, False, False, True],
                [False, False, False, False, False, True],
                [False, False, False, False, False, True],
                [False, False, False, False, False, True],
                [False, False, False, False, False, True],
                [False, False, False, False, False, True],
            ],
            index=index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "sample_0"), (0, "sample_1"), (1, "sample_0"),
                 (1, "sample_1"), (2, "sample_0"), (2, "sample_1")],
                names=["split", "sample"],
            ),
        ),
    )

def test_get_split_coverage():
    splitter = BaseModel(
        index,
        [
            [slice(5, 15), slice(10, 20)],
            [slice(10, 20), slice(15, 25)],
            [slice(15, 25), slice(20, None)],
        ],
    )
    assert_series_equal(
        splitter.get_split_coverage(normalize=False, overlapping=False),
        pd.Series(
            [15, 15, 16],
            index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
            name="split_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_split_coverage(
            normalize=True, relative=False, overlapping=False),
        pd.Series(
            [0.4838709677419355, 0.4838709677419355, 0.5161290322580645],
            index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
            name="split_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_split_coverage(
            normalize=True, relative=True, overlapping=False),
        pd.Series(
            [0.5769230769230769, 0.5769230769230769, 0.6153846153846154],
            index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
            name="split_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_split_coverage(normalize=False, overlapping=True),
        pd.Series(
            [5, 5, 5],
            index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
            name="split_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_split_coverage(normalize=True, overlapping=True),
        pd.Series(
            [0.3333333333333333, 0.3333333333333333, 0.3125],
            index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
            name="split_coverage",
        ),
    )

def test_get_sample_coverage():
    splitter = BaseModel(
        index,
        [
            [slice(5, 15), slice(10, 20)],
            [slice(10, 20), slice(15, 25)],
            [slice(15, 25), slice(20, None)],
        ],
    )
    assert_series_equal(
        splitter.get_sample_coverage(normalize=False, overlapping=False),
        pd.Series(
            [20, 21],
            index=pd.Index(["sample_0", "sample_1"],
                           dtype="object", name="sample"),
            name="sample_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_sample_coverage(
            normalize=True,
            relative=False,
            overlapping=False
        ),
        pd.Series(
            [0.6451612903225806, 0.6774193548387096],
            index=pd.Index(["sample_0", "sample_1"],
                           dtype="object", name="sample"),
            name="sample_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_sample_coverage(
            normalize=True, relative=True, overlapping=False),
        pd.Series(
            [0.7692307692307693, 0.8076923076923077],
            index=pd.Index(["sample_0", "sample_1"],
                           dtype="object", name="sample"),
            name="sample_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_sample_coverage(normalize=False, overlapping=True),
        pd.Series(
            [10, 10],
            index=pd.Index(["sample_0", "sample_1"],
                           dtype="object", name="sample"),
            name="sample_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_sample_coverage(normalize=True, overlapping=True),
        pd.Series(
            [0.5, 0.47619047619047616],
            index=pd.Index(["sample_0", "sample_1"],
                           dtype="object", name="sample"),
            name="sample_coverage",
        ),
    )

def test_get_period_coverage():
    splitter = BaseModel(
        index,
        [
            [slice(5, 5), slice(5, 6)],
            [slice(6, 8), slice(8, 11)],
            [slice(11, 15), slice(15, 20)],
            [slice(20, 26), slice(26, None)],
        ],
        allow_zero_len=True,
    )
    assert_series_equal(
        splitter.get_period_coverage(normalize=False),
        pd.Series(
            [0, 1, 2, 3, 4, 5, 6, 5],
            index=pd.MultiIndex.from_tuples(
                [
                    (0, "sample_0"),
                    (0, "sample_1"),
                    (1, "sample_0"),
                    (1, "sample_1"),
                    (2, "sample_0"),
                    (2, "sample_1"),
                    (3, "sample_0"),
                    (3, "sample_1"),
                ],
                names=["split", "sample"],
            ),
            name="period_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_period_coverage(normalize=True, relative=False),
        pd.Series(
            [
                0.0,
                0.03225806451612903,
                0.06451612903225806,
                0.0967741935483871,
                0.12903225806451613,
                0.16129032258064516,
                0.1935483870967742,
                0.16129032258064516,
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    (0, "sample_0"),
                    (0, "sample_1"),
                    (1, "sample_0"),
                    (1, "sample_1"),
                    (2, "sample_0"),
                    (2, "sample_1"),
                    (3, "sample_0"),
                    (3, "sample_1"),
                ],
                names=["split", "sample"],
            ),
            name="period_coverage",
        ),
    )
    assert_series_equal(
        splitter.get_period_coverage(normalize=True, relative=True),
        pd.Series(
            [
                0.0,
                1.0,
                0.4,
                0.6,
                0.4444444444444444,
                0.5555555555555556,
                0.5454545454545454,
                0.45454545454545453
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    (0, "sample_0"),
                    (0, "sample_1"),
                    (1, "sample_0"),
                    (1, "sample_1"),
                    (2, "sample_0"),
                    (2, "sample_1"),
                    (3, "sample_0"),
                    (3, "sample_1"),
                ],
                names=["split", "sample"],
            ),
            name="period_coverage",
        ),
    )

def test_get_coverage():
    splitter = BaseModel(
        index,
        [
            [slice(5, 15), slice(10, 20)],
            [slice(10, 20), slice(15, 25)],
            [slice(15, 25), slice(20, None)],
        ],
    )
    assert splitter.get_coverage(
        normalize=False,
        overlapping=False
    ) == 26
    assert splitter.get_coverage(
        normalize=True,
        overlapping=False
    ) == 0.8387096774193549
    assert splitter.get_coverage(
        normalize=False,
        overlapping=True
    ) == 15
    assert splitter.get_coverage(
        normalize=True,
        overlapping=True
    ) == 0.5769230769230769

def test_get_overlap_matrix():
    splitter = BaseModel(
        index,
        [
            [slice(5, 15), slice(10, 20)],
            [slice(10, 20), slice(15, 25)],
            [slice(15, 25), slice(20, None)],
        ],
    )
    np.testing.assert_array_equal(
        splitter.get_overlap_matrix(by="split", normalize=False).values,
        np.array([[15, 10, 5], [10, 15, 10], [5, 10, 16]]),
    )
    np.testing.assert_array_equal(
        splitter.get_overlap_matrix(by="split").values,
        np.array(
            [
                [1.0, 0.5, 0.19230769230769232],
                [0.5, 1.0, 0.47619047619047616],
                [0.19230769230769232, 0.47619047619047616, 1.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        splitter.get_overlap_matrix(by="sample", normalize=False).values,
        np.array([[20, 15], [15, 21]]),
    )
    np.testing.assert_array_equal(
        splitter.get_overlap_matrix(by="sample").values,
        np.array([[1.0, 0.5769230769230769], [0.5769230769230769, 1.0]]),
    )
    np.testing.assert_array_equal(
        splitter.get_overlap_matrix(by="period", normalize=False).values,
        np.array(
            [
                [10, 5, 5, 0, 0, 0],
                [5, 10, 10, 5, 5, 0],
                [5, 10, 10, 5, 5, 0],
                [0, 5, 5, 10, 10, 5],
                [0, 5, 5, 10, 10, 5],
                [0, 0, 0, 5, 5, 11],
            ]
        ),
    )
    np.testing.assert_array_equal(
        splitter.get_overlap_matrix(by="period").values,
        np.array(
            [
                [1.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0],
                [0.3333333333333333, 1.0, 1.0,
                    0.3333333333333333, 0.3333333333333333, 0.0],
                [0.3333333333333333, 1.0, 1.0,
                    0.3333333333333333, 0.3333333333333333, 0.0],
                [0.0, 0.3333333333333333, 0.3333333333333333, 1.0, 1.0, 0.3125],
                [0.0, 0.3333333333333333, 0.3333333333333333, 1.0, 1.0, 0.3125],
                [0.0, 0.0, 0.0, 0.3125, 0.3125, 1.0],
            ]
        ),
    )
