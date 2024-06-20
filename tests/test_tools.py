from collections import namedtuple
import pandas as pd
import pytest

from fold import tools

index = pd.date_range("2020-01-01", "2020-02-01", inclusive="left")


def test_replace():
    assert tools.Replace(
        "$hello$world",
        {"hello": 100}
    ).substitute(
        {"world": 300}
    ) == "100300"
    assert tools.Replace(
        "$hello$world",
        {"hello": 100}
    ).substitute(
        {"hello": 200, "world": 300}
    ) == "200300"


def test_key():
    assert tools.Key(
        "hello",
        {"hello": 100}
    ).substitute() == 100
    assert tools.Key(
        "hello",
        {"hello": 100}
    ).substitute(
        {"hello": 200}
    ) == 200


def test_lambda():
    assert tools.Lambda(
        "hello == 100",
        {"hello": 100}
    ).substitute()
    assert not tools.Lambda(
        "hello == 100",
        {"hello": 100}
    ).substitute(
        {"hello": 200}
    )


def test_function():
    assert tools.Function(
        lambda hello: hello == 100,
        {"hello": 100}
    ).substitute()
    assert not tools.Function(
        lambda hello: hello == 100,
        {"hello": 100}
    ).substitute(
        {"hello": 200}
    )


def test_substitute():
    assert tools.substitute(
        tools.Key("hello"),
        {"hello": 100}
    ) == 100
    with pytest.raises(Exception):
        tools.substitute(
            tools.Key("hello2"),
            {"hello": 100}
        )
    assert isinstance(
        tools.substitute(
            tools.Key("hello2"),
            {"hello": 100},
            strict=False
        ),
        tools.Key
    )
    assert tools.substitute(
        tools.Replace("$hello"), {"hello": 100}
    ) == "100"
    with pytest.raises(Exception):
        tools.substitute(
            tools.Replace("$hello2"),
            {"hello": 100}
        )

    assert tools.substitute(
        [tools.Key("hello")],
        {"hello": 100},
        excl_types=()
    ) == [100]

    assert tools.substitute(
        {tools.Key("hello")},
        {"hello": 100},
        excl_types=()
    ) == {100}

    assert tools.substitute(
        [tools.Key("hello")],
        {"hello": 100},
        excl_types=False
    ) == [100]

    assert tools.substitute(
        {tools.Key("hello")},
        {"hello": 100},
        excl_types=False
    ) == {100}

    assert tools.substitute(
        [tools.Key("hello")],
        {"hello": 100},
        incl_types=list
    ) == [100]

    assert tools.substitute(
        {tools.Key("hello")},
        {"hello": 100},
        incl_types=set
    ) == {100}

    assert tools.substitute(
        [tools.Key("hello")],
        {"hello": 100},
        incl_types=True
    ) == [100]

    assert tools.substitute(
        {tools.Key("hello")},
        {"hello": 100},
        incl_types=True
    ) == {100}

    assert tools.substitute(
        {"test": tools.Key("hello")},
        {"hello": 100}
    ) == {"test": 100}

    Tup = namedtuple("Tup", ["a"])
    tup = Tup(tools.Key("hello"))
    assert tools.substitute(
        tup,
        {"hello": 100}
    ) == Tup(100)

    assert tools.substitute(
        tools.Lambda("100"),
        max_depth=0
    ) == 100

    assert tools.substitute(
        (tools.Lambda("100"),),
        max_depth=0
    ) == (tools.Lambda("100"),)

    assert tools.substitute(
        (tools.Lambda("100"),),
        max_depth=1
    ) == (100,)

    assert tools.substitute(
        (tools.Lambda("100"),),
        max_len=1
    ) == (100,)

    assert tools.substitute(
        (0, tools.Lambda("100")),
        max_len=1
    ) == (
        0,
        tools.Lambda("100"),
    )

    assert tools.substitute(
        (0, tools.Lambda("100")),
        max_len=2
    ) == (
        0,
        100,
    )


def test_get_func_arg_names():
    def f(a, *args, b=2, **kwargs):
        pass

    assert tools.Function.get_func_arg_names(f) == ["a", "b"]


def test_relative_period():
    assert tools.RelativePeriod().to_slice(30) == slice(0, 30)
    assert tools.RelativePeriod(
        offset=1
    ).to_slice(
        30
    ) == slice(1, 30)
    assert tools.RelativePeriod(
        offset=0.5
    ).to_slice(
        30
    ) == slice(15, 30)
    assert tools.RelativePeriod(
        offset_anchor="end",
        offset=-1.0
    ).to_slice(
        30
    ) == slice(0, 30)
    assert tools.RelativePeriod(
        offset_anchor="prev_start"
    ).to_slice(
        30, prev_start=1
    ) == slice(1, 30)
    assert tools.RelativePeriod(
        offset_anchor="prev_end"
    ).to_slice(
        30, prev_end=1
    ) == slice(1, 30)
    assert tools.RelativePeriod(
        offset_anchor="prev_end",
        offset=0.5,
        offset_space="free"
    ).to_slice(
        30, prev_end=10
    ) == slice(20, 30)
    assert tools.RelativePeriod(
        offset_anchor="prev_end",
        offset=-0.5,
        offset_space="free"
    ).to_slice(
        30, prev_end=10
    ) == slice(5, 30)
    assert tools.RelativePeriod(
        offset_anchor="prev_end",
        offset=0.5,
        offset_space="all"
    ).to_slice(
        30, prev_end=10
    ) == slice(25, 30)
    assert tools.RelativePeriod(
        length=10
    ).to_slice(
        30
    ) == slice(0, 10)
    assert tools.RelativePeriod(
        length=0.5
    ).to_slice(
        30
    ) == slice(0, 15)
    assert tools.RelativePeriod(
        offset_anchor="prev_end",
        length=10
    ).to_slice(
        30, prev_end=10
    ) == slice(10, 20)
    assert tools.RelativePeriod(
        offset_anchor="prev_end",
        length=0.5
    ).to_slice(
        30, prev_end=10
    ) == slice(10, 20)
    assert tools.RelativePeriod(
        offset_anchor="end",
        length=-0.5
    ).to_slice(
        30, prev_end=0
    ) == slice(15, 30)
    assert tools.RelativePeriod(
        offset_anchor="end",
        length=-0.5
    ).to_slice(
        30, prev_end=10
    ) == slice(15, 30)
    assert tools.RelativePeriod(
        offset_anchor="end",
        length=-0.5,
        length_space="free_or_prev"
    ).to_slice(
        30,
        prev_end=10
    ) == slice(20, 30)
    assert tools.RelativePeriod(
        offset_anchor="prev_end",
        length=-0.5
    ).to_slice(
        30,
        prev_end=10
    ) == slice(5, 10)
    assert tools.RelativePeriod(
        offset_anchor="prev_end",
        length=0.5,
        length_space="all"
    ).to_slice(
        30,
        prev_end=10
    ) == slice(10, 25)
    assert tools.RelativePeriod(
        offset=-10,
        length=50
    ).to_slice(
        30
    ) == slice(0, 30)
    assert tools.RelativePeriod(
        length=index[5] - index[0]
    ).to_slice(
        len(index),
        index=index
    ) == tools.RelativePeriod(
        length=5
    ).to_slice(
        len(index),
        index=index
    )
    assert tools.RelativePeriod(
        offset=1,
        length=index[5] - index[0]
    ).to_slice(
        len(index),
        index=index
    ) == tools.RelativePeriod(
        offset=1,
        length=5
    ).to_slice(
        len(index), index=index
    )
    assert tools.RelativePeriod(
        offset=index[1] - index[0],
        length=index[5] - index[0]
    ).to_slice(
        len(index), index=index
    ) == tools.RelativePeriod(
        offset=1,
        length=5
    ).to_slice(
        len(index), index=index
    )
    assert tools.RelativePeriod(
        offset_anchor="end",
        length=index[0] - index[5]
    ).to_slice(
        len(index), index=index
    ) == tools.RelativePeriod(
        offset_anchor="end",
        length=-5
    ).to_slice(
        len(index), index=index
    )
    assert tools.RelativePeriod(
        offset="-3 days",
        length="5 days"
    ).to_slice(
        len(index), index=index
    ) == tools.RelativePeriod(
        offset=-3,
        length=5
    ).to_slice(
        len(index), index=index
    )
    assert tools.RelativePeriod(
        offset="3 days",
        length="-5 days"
    ).to_slice(
        len(index), index=index
    ) == tools.RelativePeriod(
        offset=3,
        length=-5
    ).to_slice(
        len(index),
        index=index
    )
    assert tools.RelativePeriod(
        offset="-3 days",
        offset_anchor="end",
        length=index[5] - index[0]
    ).to_slice(
        len(index), index=index
    ) == tools.RelativePeriod(
        offset=-3,
        offset_anchor="end",
        length=5
    ).to_slice(
        len(index), index=index
    )
    assert tools.RelativePeriod(
        offset="3 days",
        offset_anchor="end",
        length=index[0] - index[5]
    ).to_slice(
        len(index), index=index
    ) == tools.RelativePeriod(
        offset=3,
        offset_anchor="end",
        length=-5
    ).to_slice(
        len(index),
        index=index
    )
    with pytest.raises(Exception):
        tools.RelativePeriod(
            offset=-10, length=50, out_of_bounds="raise"
        ).to_slice(
            30
        )
    with pytest.raises(Exception):
        tools.RelativePeriod(offset_anchor="hello")
    with pytest.raises(Exception):
        tools.RelativePeriod(offset_space="hello")
    with pytest.raises(Exception):
        tools.RelativePeriod(length_space="hello")
    with pytest.raises(Exception):
        tools.RelativePeriod(out_of_bounds="hello")
