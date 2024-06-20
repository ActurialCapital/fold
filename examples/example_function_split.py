from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

length = 5000
n_paths = 10

end_date = datetime.now().date()

start_date = end_date - timedelta(days=length - 1)

data = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=[f'path_{n}' for n in range(1, n_paths + 1)],
    index=pd.date_range(start=start_date, end=end_date, freq="D")
)


if __name__ == '__main__':

    from fold import FunctionSplit, Key

    # Example 1
    # ---------

    def split_func(index, prev_start):
        if prev_start is None:
            prev_start = index[0]
        new_start = prev_start + pd.offsets.MonthBegin(1)
        new_end = new_start + pd.DateOffset(years=1)
        if new_end > index[-1] + index.freq:
            return None
        return [
            slice(new_start, new_start + pd.offsets.MonthBegin(9)),
            slice(new_start + pd.offsets.MonthBegin(9), new_end)
        ]
    
    model = FunctionSplit(
        data.index,
        split_func=split_func,
        split_args=(Key("index"), Key("prev_start")),
        index_bounds=True,
        fix_ranges=True,
        sample_labels=["IS", "OOS"],
    )

    # Base
    print(model.splits_arr)
    print(model.splits_arr.dtype)
    print(model.split_labels)
    print(model.sample_labels)
    print(model.splits)
    print(model.n_splits)
    print(model.n_samples)
    print(model.index)
    print(model.split_index())
    print(model.sample_index())
    print(model.select_period())

    # Bounds
    print(model.get_bounds_arr())
    print(model.get_bounds(index_bounds=False))
    print(model.get_bounds(index_bounds=True))
    print(model.index_bounds)

    # Duration
    print(model.get_duration())
    print(model.get_duration(index_bounds=True))
    print(model.duration)
    print(model.index_duration)

    # Masks
    mask = model.get_mask()
    print(mask)
    print(mask["2021":"2021"].any())
    print(mask.resample("YS").sum())
    results = []
    for mask in model.get_iter_split_masks():
        results.append(mask.resample("YS").sum())
    print(pd.concat(results, axis=1, keys=model.split_labels))

    # Coverage
    print(model.get_coverage())
    print(model.get_coverage(overlapping=True))

    # Split coverage
    print(model.get_split_coverage())
    print(model.get_split_coverage(overlapping=True))

    # Set coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Range coverage
    print(model.get_period_coverage())
    print(model.get_period_coverage(relative=True))

    # Overlap
    print(model.get_overlap_matrix(by="period", normalize=False))

    # Grouping
    print(model.get_bounds(index_bounds=True))

    # Train test split
    slices = model.train_test_split(data)
    print(slices)

    # Example 2
    # ---------

    def offset(params: tuple):
        """Determine the offset based on parameters."""
        period, freq = params
        if freq.startswith(('week', 'W', 'w')):
            return pd.offsets.Week(period)
        elif freq.startswith(('month', 'M', 'm')):
            return pd.offsets.MonthBegin(period)
        elif freq.startswith(('quarter', 'Q', 'q')):
            return pd.offsets.QuarterBegin(period, startingMonth=1)
        elif freq.startswith(('year', 'Y', 'y')):
            return pd.DateOffset(years=period)
        else:
            raise ValueError('Invalid frequency.')
    
    def split_func(
        n_train: tuple,
        n_test: tuple,
        index: pd.DatetimeIndex,
        prev_start: pd.Timestamp
    ):
        """Define the split function for creating training and testing periods.

        Parameters
        ----------
        index : pd.DatetimeIndex
            The index of the time series data.
        prev_start : pd.Timestamp
            The start date of the previous split.

        """
        # If this is the first split, prev_start (i.e., the start index of the
        # previous split) will be None
        if prev_start is None:
            prev_start = index[0]
        # The start date of this window is the beginning of the next quarter
        # (inclusive) - For next month, use MonthBegin() instead.
        # Testset increment
        # pd.offsets.QuarterBegin(n_test)
        new_start = prev_start + offset(n_test)
        # Total number of years to split
        # The end date of this window is the same date but in the n_train + n_test
        # year (exclusive)
        # pd.DateOffset(years=n_train + n_test)
        new_end = new_start + offset(n_train) + offset(n_test)
        # If the split is incomplete (i.e., the end date comes after the next
        # possible end date), abort!
        if new_end > index[-1] + index.freq:
            return None
        # Trainset increment
        # pd.DateOffset(years=n_train)
        offset_period = new_start + offset(n_train)
        return [
            # Trainset increment
            # Allocate n_train freq for the IS period and n_test freq for
            # the OOS period
            slice(new_start, offset_period),
            slice(offset_period, new_end)
        ]

    model = FunctionSplit(
        data.index,
        split_func=split_func,
        # Replace strings by the respective context
        split_args=(
            (5, 'Y'),
            (1, 'Y'),
            Key("index"),
            Key("prev_start")
        ),
        # Return the bounds of each range as timestamp, not
        # number of bars, otherwise prev_start will be an integer
        index_bounds=True,
        sample_labels=["IS", "OOS"]
    )

    # Base
    print(model.splits_arr)
    print(model.splits_arr.dtype)
    print(model.split_labels)
    print(model.sample_labels)
    print(model.splits)
    print(model.n_splits)
    print(model.n_samples)
    print(model.index)
    print(model.split_index())
    print(model.sample_index())
    print(model.select_period())

    # Bounds
    print(model.get_bounds_arr())
    print(model.get_bounds(index_bounds=False))
    print(model.get_bounds(index_bounds=True))
    print(model.index_bounds)

    # Duration
    print(model.get_duration())
    print(model.get_duration(index_bounds=True))
    print(model.duration)
    print(model.index_duration)

    # Masks
    mask = model.get_mask()
    print(mask)
    print(mask["2021":"2021"].any())
    print(mask.resample("YS").sum())
    results = []
    for mask in model.get_iter_split_masks():
        results.append(mask.resample("YS").sum())
    print(pd.concat(results, axis=1, keys=model.split_labels))

    # Coverage
    print(model.get_coverage())
    print(model.get_coverage(overlapping=True))

    # Split coverage
    print(model.get_split_coverage())
    print(model.get_split_coverage(overlapping=True))

    # Set coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Range coverage
    print(model.get_period_coverage())
    print(model.get_period_coverage(relative=True))

    # Overlap
    print(model.get_overlap_matrix(by="period", normalize=False))

    # Grouping
    print(model.get_bounds(index_bounds=True))

    # Train test split
    slices = model.train_test_split(data)
    print(slices)
    
    # Example 3
    # ---------

    def split_func(splits, bounds, index):
        if len(splits) == 0:
            new_split = (slice(0, 20), slice(20, 30))
        else:
            # Previous split, first set, right bound
            prev_end = bounds[-1][0][1]
            new_split = (
                slice(prev_end, prev_end + 20),
                slice(prev_end + 20, prev_end + 30)
            )
        if new_split[-1].stop > len(index):
            return None
        return new_split

    model = FunctionSplit(
        data.index,
        split_func,
        split_args=(
            Key("splits"),
            Key("bounds"),
            Key("index"),
        ),
        sample_labels=["IS", "OOS"]
    )

    # Base
    print(model.splits_arr)
    print(model.splits_arr.dtype)
    print(model.split_labels)
    print(model.sample_labels)
    print(model.splits)
    print(model.n_splits)
    print(model.n_samples)
    print(model.index)
    print(model.split_index())
    print(model.sample_index())
    print(model.select_period())

    # Bounds
    print(model.get_bounds_arr())
    print(model.get_bounds(index_bounds=False))
    print(model.get_bounds(index_bounds=True))
    print(model.index_bounds)

    # Duration
    print(model.get_duration())
    print(model.get_duration(index_bounds=True))
    print(model.duration)
    print(model.index_duration)

    # Masks
    mask = model.get_mask()
    print(mask)
    print(mask["2021":"2021"].any())
    print(mask.resample("YS").sum())
    results = []
    for mask in model.get_iter_split_masks():
        results.append(mask.resample("YS").sum())
    print(pd.concat(results, axis=1, keys=model.split_labels))

    # Coverage
    print(model.get_coverage())
    print(model.get_coverage(overlapping=True))

    # Split coverage
    print(model.get_split_coverage())
    print(model.get_split_coverage(overlapping=True))

    # Set coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Range coverage
    print(model.get_period_coverage())
    print(model.get_period_coverage(relative=True))

    # Overlap
    print(model.get_overlap_matrix(by="period", normalize=False))

    # Grouping
    print(model.get_bounds(index_bounds=True))

    # Train test split
    slices = model.train_test_split(data)
    print(slices)
