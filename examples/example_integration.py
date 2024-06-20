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

    from sklearn.model_selection import TimeSeriesSplit
    from fold import SklearnFold, WrapperSplit

    # Example 1
    # ---------

    model = SklearnFold(
        data.index,
        sk_model=TimeSeriesSplit(n_splits=2),
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

    # Example 2
    # ---------

    from fold import RollingSplit

    wrapper = WrapperSplit(
        model=RollingSplit,
        length=360,
        split=0.5,
        offset_anchor_set=None,
        sample_labels=["IS", "OOS"]
    )

    # Number of splits
    print(wrapper.get_n_splits(data))

    # Check Splitter
    model = wrapper.get_model(data.index)

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

    from fold import GrouperSplit

    wrapper = WrapperSplit(
        model=GrouperSplit,
        by="YS",
        split=0.5,
        sample_labels=["IS", "OOS"]
    )

    # Number of splits
    print(wrapper.get_n_splits(data))

    # Check Splitter
    model = wrapper.get_model(data.index)

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

    from fold import ExpandingSplit

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])

    wrapper = WrapperSplit(
        model=ExpandingSplit,
        min_length=2,
        offset=1,
        split=-1,
        sample_labels=["IS", "OOS"]
    )

    # Check CV generator
    for i, (train, test) in enumerate(wrapper.split(X)):
        print("Split %d:" % i)
        X_train, X_test = X[train], X[test]
        print("  X:", X_train.tolist(), X_test.tolist())
        y_train, y_test = y[train], y[test]
        print("  y:", y_train.tolist(), y_test.tolist())
