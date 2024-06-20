from functools import partial
import numpy as np
import pandas as pd
import pytest

from fold.utils.grouper import CustomGrouper


assert_index_equal = partial(
    pd.testing.assert_index_equal,
    rtol=1e-06,
    atol=0
)


grouped_index = pd.MultiIndex.from_arrays(
    [
        [1, 1, 1, 1, 0, 0, 0, 0],
        [3, 3, 2, 2, 1, 1, 0, 0],
        [7, 6, 5, 4, 3, 2, 1, 0]
    ],
    names=["first", "second", "third"],
)


def test_group_by_to_index():
    assert not CustomGrouper.group_by_to_index(
        grouped_index,
        group_by=False
    )
    assert CustomGrouper.group_by_to_index(
        grouped_index,
        group_by=None
    ) is None
    assert_index_equal(
        CustomGrouper.group_by_to_index(grouped_index, group_by=True),
        pd.Index(["group"] * len(grouped_index), name="group"),
    )
    assert_index_equal(
        CustomGrouper.group_by_to_index(grouped_index, group_by=0),
        pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
    )
    assert_index_equal(
        CustomGrouper.group_by_to_index(grouped_index, group_by="first"),
        pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
    )
    assert_index_equal(
        CustomGrouper.group_by_to_index(grouped_index, group_by=[0, 1]),
        pd.MultiIndex.from_tuples(
            [(1, 3), (1, 3), (1, 2), (1, 2), (0, 1), (0, 1), (0, 0), (0, 0)],
            names=["first", "second"],
        ),
    )
    assert_index_equal(
        CustomGrouper.group_by_to_index(
            grouped_index, group_by=["first", "second"]),
        pd.MultiIndex.from_tuples(
            [(1, 3), (1, 3), (1, 2), (1, 2), (0, 1), (0, 1), (0, 0), (0, 0)],
            names=["first", "second"],
        ),
    )
    assert_index_equal(
        CustomGrouper.group_by_to_index(
            grouped_index,
            group_by=np.array([3, 2, 1, 1, 1, 0, 0, 0])
        ),
        pd.Index([3, 2, 1, 1, 1, 0, 0, 0], dtype="int64", name="group"),
    )
    assert_index_equal(
        CustomGrouper.group_by_to_index(
            grouped_index,
            group_by=pd.Index([3, 2, 1, 1, 1, 0, 0, 0], name="fourth"),
        ),
        pd.Index([3, 2, 1, 1, 1, 0, 0, 0], dtype="int64", name="fourth"),
    )

def test_get_groups_and_index():
    a, b = CustomGrouper.group_by_to_groups_and_index(
        grouped_index,
        group_by=None
    )
    np.testing.assert_array_equal(a, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    assert_index_equal(b, grouped_index)
    a, b = CustomGrouper.group_by_to_groups_and_index(
        grouped_index,
        group_by=0
    )
    np.testing.assert_array_equal(a, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    assert_index_equal(b, pd.Index([1, 0], dtype="int64", name="first"))
    a, b = CustomGrouper.group_by_to_groups_and_index(
        grouped_index,
        group_by=[0, 1]
    )
    np.testing.assert_array_equal(a, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
    assert_index_equal(
        b,
        pd.MultiIndex.from_tuples(
            [(1, 3), (1, 2), (0, 1), (0, 0)],
            names=["first", "second"]
        ),
    )

def test_is_grouped():
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouped()
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouped(group_by=True)
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouped(group_by=1)
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouped(group_by=False)
    assert not CustomGrouper(
        grouped_index
    ).is_grouped()
    assert CustomGrouper(
        grouped_index
    ).is_grouped(group_by=0)
    assert CustomGrouper(
        grouped_index
    ).is_grouped(group_by=True)
    assert not CustomGrouper(
        grouped_index
    ).is_grouped(group_by=False)
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouped(
        group_by=grouped_index.get_level_values(0) + 1
    )  # only labels

def test_is_grouping_enabled():
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_enabled()
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_enabled(group_by=True)
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_enabled(group_by=1)
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_enabled(group_by=False)
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_enabled()
    assert CustomGrouper(
        grouped_index
    ).is_grouping_enabled(group_by=0)
    assert CustomGrouper(
        grouped_index
    ).is_grouping_enabled(group_by=True)
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_enabled(group_by=False)
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_enabled(
        group_by=grouped_index.get_level_values(0) + 1
    )  # only labels

def test_is_grouping_disabled():
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_disabled()
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_disabled(group_by=True)
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_disabled(group_by=1)
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_disabled(group_by=False)
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_disabled()
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_disabled(group_by=0)
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_disabled(group_by=True)
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_disabled(group_by=False)
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_disabled(
        group_by=grouped_index.get_level_values(0) + 1
    )  # only labels

def test_is_grouping_modified():
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_modified()
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_modified(group_by=True)
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_modified(group_by=1)
    assert CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_modified(group_by=False)
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_modified()
    assert CustomGrouper(
        grouped_index
    ).is_grouping_modified(group_by=0)
    assert CustomGrouper(
        grouped_index
    ).is_grouping_modified(group_by=True)
    assert not CustomGrouper(
        grouped_index
    ).is_grouping_modified(group_by=False)
    assert not CustomGrouper(
        grouped_index,
        group_by=0
    ).is_grouping_modified(
        group_by=grouped_index.get_level_values(0) + 1
    )  # only labels

def test_check_group_by():
    CustomGrouper(
        grouped_index,
        group_by=None,
        allow_enable=True
    ).check_group_by(group_by=0)
    with pytest.raises(Exception):
        CustomGrouper(
            grouped_index,
            group_by=None,
            allow_enable=False
        ).check_group_by(group_by=0)
    CustomGrouper(
        grouped_index,
        group_by=0,
        allow_disable=True
    ).check_group_by(group_by=False)
    with pytest.raises(Exception):
        CustomGrouper(
            grouped_index,
            group_by=0,
            allow_disable=False
        ).check_group_by(group_by=False)
    CustomGrouper(
        grouped_index,
        group_by=0,
        allow_modify=True
    ).check_group_by(group_by=1)
    CustomGrouper(
        grouped_index,
        group_by=0,
        allow_modify=False
    ).check_group_by(
        group_by=np.array([2, 2, 2, 2, 3, 3, 3, 3]),
    )
    with pytest.raises(Exception):
        CustomGrouper(
            grouped_index,
            group_by=0,
            allow_modify=False
        ).check_group_by(group_by=1)

def test_resolve_group_by():
    assert CustomGrouper(
        grouped_index,
        group_by=None
    ).resolve_group_by() is None  # default
    assert_index_equal(
        CustomGrouper(
            grouped_index,
            group_by=None
        ).resolve_group_by(
            group_by=0
        ),  # overrides
        pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
    )
    assert_index_equal(
        CustomGrouper(
            grouped_index,
            group_by=0
        ).resolve_group_by(),  # default
        pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
    )
    assert_index_equal(
        CustomGrouper(
            grouped_index,
            group_by=0
        ).resolve_group_by(
            group_by=1
        ),  # overrides
        pd.Index([3, 3, 2, 2, 1, 1, 0, 0], dtype="int64", name="second"),
    )

def test_get_groups():
    np.testing.assert_array_equal(
        CustomGrouper(grouped_index).get_groups(),
        np.array([0, 1, 2, 3, 4, 5, 6, 7]),
    )
    np.testing.assert_array_equal(
        CustomGrouper(grouped_index).get_groups(group_by=0),
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    )

def test_get_index():
    assert_index_equal(
        CustomGrouper(grouped_index).get_index(),
        CustomGrouper(grouped_index).index,
    )
    assert_index_equal(
        CustomGrouper(grouped_index).get_index(group_by=0),
        pd.Index([1, 0], dtype="int64", name="first"),
    )

def test_iter_group_idxs():
    np.testing.assert_array_equal(
        np.concatenate(
            tuple(CustomGrouper(grouped_index).iter_group_idxs())
        ),
        np.array([0, 1, 2, 3, 4, 5, 6, 7]),
    )
    np.testing.assert_array_equal(
        np.concatenate(
            tuple(CustomGrouper(grouped_index).iter_group_idxs(group_by=0))
        ),
        np.array([0, 1, 2, 3, 4, 5, 6, 7]),
    )
