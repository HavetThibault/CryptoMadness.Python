import numpy as np

from crypto.analyse.labels_and_preds_crypto_accuracy import LabelsAndPredsCryptoAccuracy
from crypto.mlp_experiences.regression1_params import Regression1Params
from helper_sdk.pandas_helper import df_min_max
from ml_sdk.dataset.cook.ds_source_file import read_dataset

if __name__ == '__main__':
    val_filepath = Regression1Params.VAL_PATH
    src_df = read_dataset(Regression1Params.SRC_DS_PATH, header=0)
    close_cols = []
    for col in list(src_df.columns):
        if col.startswith('Close'):
            close_cols.append(col)
    close_min, close_max = df_min_max(src_df[close_cols])

    val_df = read_dataset(val_filepath, header=0)
    last_close_col = LabelsAndPredsCryptoAccuracy.get_last_close_col(list(val_df.columns))
    diffs = []
    for row in iter(val_df[[last_close_col, 'FutureClose']].iloc):
        last_close = row[last_close_col] * (close_max - close_min) + close_min
        diffs.append((row['FutureClose'] - last_close) * (row['FutureClose'] - last_close))
    # 4163
    print(f'Std of the differences is : {np.mean(diffs)}')