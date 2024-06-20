from functools import partial
import pandas as pd

from fold.utils import indexing


assert_index_equal = partial(
    pd.testing.assert_index_equal,
    rtol=1e-06,
    atol=0
)


# Initialize global variables
ser1 = pd.Series([1])
ser2 = pd.Series(
    [1, 2, 3],
    index=pd.Index(["x2", "y2", "z2"], name="i2"),
    name="a2"
)

df1 = pd.DataFrame([[1]])
df2 = pd.DataFrame(
    [[1]],
    index=pd.Index(["x3"], name="i3"),
    columns=pd.Index(["a3"], name="c3")
)
df3 = pd.DataFrame(
    [[1], [2], [3]],
    index=pd.Index(["x4", "y4", "z4"], name="i4"),
    columns=pd.Index(["a4"], name="c4")
)
multi_index = pd.MultiIndex.from_arrays(
    [["x7", "y7", "z7"], ["x8", "y8", "z8"]],
    names=["i7", "i8"]
)
multi_column = pd.MultiIndex.from_arrays(
    [["a7", "b7", "c7"], ["a8", "b8", "c8"]],
    names=["c7", "c8"]
)
df4 = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    index=multi_index,
    columns=multi_column
)


def test_repeat_index():
    i = pd.Index([1, 2, 3], name="i")
    assert_index_equal(
        indexing.repeat_index(i, 3),
        pd.Index([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype="int64", name="i"),
    )
    assert_index_equal(
        indexing.repeat_index(multi_index, 3),
        pd.MultiIndex.from_tuples(
            [
                ("x7", "x8"),
                ("x7", "x8"),
                ("x7", "x8"),
                ("y7", "y8"),
                ("y7", "y8"),
                ("y7", "y8"),
                ("z7", "z8"),
                ("z7", "z8"),
                ("z7", "z8"),
            ],
            names=["i7", "i8"],
        ),
    )
    assert_index_equal(
        indexing.repeat_index([0], 3),
        pd.Index([0, 1, 2], dtype="int64")
    )  # empty
    assert_index_equal(
        indexing.repeat_index(ser1.index, 3),
        pd.RangeIndex(start=0, stop=3, step=1)  # simple range,
    )

def test_tile_index():
    i = pd.Index([1, 2, 3], name="i")
    assert_index_equal(
        indexing.tile_index(i, 3),
        pd.Index([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype="int64", name="i"),
    )
    assert_index_equal(
        indexing.tile_index(multi_index, 3),
        pd.MultiIndex.from_tuples(
            [
                ("x7", "x8"),
                ("y7", "y8"),
                ("z7", "z8"),
                ("x7", "x8"),
                ("y7", "y8"),
                ("z7", "z8"),
                ("x7", "x8"),
                ("y7", "y8"),
                ("z7", "z8"),
            ],
            names=["i7", "i8"],
        ),
    )
    assert_index_equal(
        indexing.tile_index([0], 3),
        pd.Index([0, 1, 2], dtype="int64")
    )  # empty
    assert_index_equal(
        indexing.tile_index(ser1.index, 3),
        pd.RangeIndex(start=0, stop=3, step=1)  # simple range,
    )

def test_stack_index():
    assert_index_equal(
        indexing.stack_index([ser2.index, df3.index, df4.index]),
        pd.MultiIndex.from_tuples(
            [("x2", "x4", "x7", "x8"),
             ("y2", "y4", "y7", "y8"),
             ("z2", "z4", "z7", "z8")],
            names=["i2", "i4", "i7", "i8"],
        ),
    )
    assert_index_equal(
        indexing.stack_index(
            [ser2.index, df3.index, ser2.index],
        ),
        pd.MultiIndex.from_tuples(
            [("x2", "x4", "x2"),
             ("y2", "y4", "y2"),
             ("z2", "z4", "z2")],
            names=["i2", "i4", "i2"],
        ),
    )
    

def test_combine_index():
    assert_index_equal(
        indexing.combine_index(
            [pd.Index([1]), pd.Index([2, 3])], 
        ),
        pd.MultiIndex.from_tuples([(1, 2), (1, 3)]),
    )
    assert_index_equal(
        indexing.combine_index(
            [pd.Index([1, 2]), pd.Index([3])], 
        ),
        pd.MultiIndex.from_tuples([(1, 3), (2, 3)]),
    )
    assert_index_equal(
        indexing.combine_index([df3.index, df4.index]),
        pd.MultiIndex.from_tuples(
            [
                ("x4", "x7", "x8"),
                ("x4", "y7", "y8"),
                ("x4", "z7", "z8"),
                ("y4", "x7", "x8"),
                ("y4", "y7", "y8"),
                ("y4", "z7", "z8"),
                ("z4", "x7", "x8"),
                ("z4", "y7", "y8"),
                ("z4", "z7", "z8"),
            ],
            names=["i4", "i7", "i8"],
        ),
    )
