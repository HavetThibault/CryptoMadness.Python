# from configparser import Error
#
# import pandas as pd
#
# from helper_sdk.pandas_helper import df_min_max
# from ml_sdk.dataset.cook.ds_source_file import shuffle_dataset
# from ml_sdk.dataset.file.model_ds_creation import create_save_train_val_ds
#
# if __name__ == '__main__':
#     inputs = 60
#     after = 2
#     version = 1
#     intervals = 0
#     normalized = 1
#     str_id = f'b{inputs}_a{after}_i{intervals}_v{version}'
#
#     root = f'C:/MyProgs/Python/CryptoMadness.Python/Data/'
#     ds_path = f'{root}CV_BTC_Data_{str_id}.csv'
#     train_path = f'{root}train_val/btc_{str_id}_n{normalized}_train.csv'
#     val_path = f'{root}train_val/btc_{str_id}_n{normalized}_val.csv'
#
#     ds = pd.read_csv(ds_path, sep=',', header=0)
#     ds = shuffle_dataset(ds)
#
#     if normalized == 1:
#         cols: list[str] = list(ds.columns)
#         vol_cols = []
#         close_cols = []
#         for col in cols:
#             if col.startswith('Volume'):
#                 vol_cols.append(col)
#             elif col.startswith('Close') and not col.endswith('prediction'):
#                 close_cols.append(col)
#         vol_min, vol_max = df_min_max(ds[vol_cols])
#         close_min, close_max = df_min_max(ds[close_cols])
#         print(f'Volume range: [{vol_min}, {vol_max}]')
#         print(f'Close range: [{close_min}, {close_max}]')
#         ds[vol_cols] = ds[vol_cols].apply(lambda a: (a - vol_min) / (vol_max - vol_min), axis="columns")
#         ds[close_cols] = ds[close_cols].apply(lambda a: (a - close_min) / (close_max - close_min), axis="columns")
#
#     create_save_train_val_ds(
#         ds,
#         0.3,
#         train_path,
#         val_path,
#         False,
#         False)
