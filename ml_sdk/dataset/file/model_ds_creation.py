import pandas as pd

from ml_sdk.dataset.cook.ds_source_file import shuffle_dataset, write_dataset


def create_save_train_val_ds(ds: pd.DataFrame, val_ds_ratio, train_label_path, val_label_path, shuffle,
                             overwrite=False) -> tuple[int, int]:
    ds_len = len(ds)
    train_ds_len = int((1 - val_ds_ratio) * ds_len)
    val_ds_len = ds_len - train_ds_len
    val_ds = ds.iloc[:val_ds_len]
    train_ds = ds.iloc[val_ds_len:]
    if shuffle:
        train_ds = shuffle_dataset(train_ds)
        val_ds = shuffle_dataset(val_ds)
    write_dataset(train_ds, train_label_path, overwrite, header=True)
    write_dataset(val_ds, val_label_path, overwrite, header=True)
    return train_ds_len, val_ds_len
