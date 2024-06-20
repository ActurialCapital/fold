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

    from fold import BaseModel, RelativePeriod

    # Example 1
    # ---------

    model = BaseModel(
        index=data.index,
        splits=[
            (
                RelativePeriod(),
                RelativePeriod(offset=0.5, length=0.25, length_space="all")
            ),
            (
                RelativePeriod(),
                RelativePeriod(offset=0.25, length=0.25, length_space="all")
            ),
            (
                RelativePeriod(),
                RelativePeriod(offset=0, length=0.25, length_space="all")
            ),
        ],
        backwards=True,
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

    range_00 = np.arange(0, 5)
    range_01 = np.arange(5, 15)
    range_10 = np.arange(15, 30)
    range_11 = np.arange(30, 50)

    model_not_fixed = BaseModel(
        data.index,
        [[range_00, range_01], [range_10, range_11]],
        fix_ranges=False
    )
    print(model_not_fixed.splits)

    model_fixed = BaseModel(
        data.index,
        [[range_00, range_01], [range_10, range_11]],
        fix_ranges=True
    )
    print(model_fixed.splits)
