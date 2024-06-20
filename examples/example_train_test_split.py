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

    from fold import train_test_split, PurgedWalkForwardSplit

    # Example 1
    # ---------

    model = train_test_split(
        obj=data,
        model=PurgedWalkForwardSplit(index=data.index),
    )
    print(model)

       

  

