import os
from typing import Optional

import pandas as pd

GLOBAL_PIC_NAME_END = 'g'
CROP_PIC_NAME_END = 'c'

EXPOSURE = 'exposure'


def read_dataset(ds_path, sep=',', header: Optional[int]=None) -> pd.DataFrame:
    return pd.read_csv(ds_path, sep=sep, header=header)


def write_dataset(ds: pd.DataFrame, path, overwrite=False, header: Optional[bool]=None):
    if os.path.exists(path) and not overwrite:
        raise Exception(f'A file already exists with path: {path}')
    ds.to_csv(path, index=False, header=header)


def shuffle_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.sample(frac=1).reset_index(drop=True)

