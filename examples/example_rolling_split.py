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

    from fold import RollingSplit

    # Example 1
    # ---------
    
    model = RollingSplit(
        data.index,
        length=360,
        split=0.5,
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

    # Sample coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Period coverage
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
    
    model = RollingSplit(
        data.index,
        length="360 days",
        split=0.5,
        freq="daily",
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

    # Sample coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Period coverage
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
    
    # Divide a range into a set of non-overlapping ranges
    model = RollingSplit(data.index, length=30)

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

    # Sample coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Period coverage
    print(model.get_period_coverage())
    print(model.get_period_coverage(relative=True))

    # Overlap
    print(model.get_overlap_matrix(by="period", normalize=False))

    # Grouping
    print(model.get_bounds(index_bounds=True))

    # Train test split
    slices = model.train_test_split(data)
    print(slices)
        

    # Example 4
    # ---------
    
    # Divide a range into ranges, each split into 1/2
    model = RollingSplit(
        data.index,
        60,
        split=1/2,
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

    # Sample coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Period coverage
    print(model.get_period_coverage())
    print(model.get_period_coverage(relative=True))

    # Overlap
    print(model.get_overlap_matrix(by="period", normalize=False))

    # Grouping
    print(model.get_bounds(index_bounds=True))

    # Train test split
    slices = model.train_test_split(data)
    print(slices)


    # Example 5
    # ---------
    
    # Make the ranges above non-overlapping by using the right bound of the
    # last set as an offset anchor
    model = RollingSplit(
        data.index,
        60,
        offset_anchor_set=-1,
        split=1/2,
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

    # Sample coverage
    print(model.get_sample_coverage())
    print(model.get_sample_coverage(relative=True))

    # Period coverage
    print(model.get_period_coverage())
    print(model.get_period_coverage(relative=True))

    # Overlap
    print(model.get_overlap_matrix(by="period", normalize=False))

    # Grouping
    print(model.get_bounds(index_bounds=True))

    # Train test split
    slices = model.train_test_split(data)
    print(slices)
