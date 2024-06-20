import numpy as np
import pandas as pd

from fold import WrapperSplit, BaseModel

index = pd.date_range("2020-01-01", "2020-02-01", inclusive="left")


def test_integration():
    model = WrapperSplit(
        BaseModel,
        splits=[
            [slice(5, 5), slice(5, 6)],
            [slice(6, 8), slice(8, 11)],
            [slice(11, 15), slice(15, 20)],
            [slice(20, 26), slice(26, None)],
        ],
        allow_zero_len=True,
    )
    
    sr = pd.Series(np.arange(len(index)), index=index)
    
    splits = list(model.split(sr))
    
    np.testing.assert_array_equal(
        splits[0][0],
        np.array([], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        splits[0][1],
        np.array([5], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        splits[1][0],
        np.array([6, 7], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        splits[1][1],
        np.array([8, 9, 10], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        splits[2][0],
        np.array([11, 12, 13, 14], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        splits[2][1],
        np.array([15, 16, 17, 18, 19], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        splits[3][0],
        np.array([20, 21, 22, 23, 24, 25], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        splits[3][1],
        np.array([26, 27, 28, 29, 30], dtype=np.int64)
    )
