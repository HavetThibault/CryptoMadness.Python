import pandas as pd

from ml_sdk.dataset.cook.ds_source_file import read_dataset
from ml_sdk.dataset.file.model_ds_creation import create_save_train_val_ds

if __name__ == '__main__':
    ds_path = 'D:/MyProg/CSharp/CryptoMadness/CV_BTC_Data.csv'
    train_path = 'C:/Users/Thibault/Downloads/BTC/data/btc_train.csv'
    val_path = 'C:/Users/Thibault/Downloads/BTC/data/btc_val.csv'

    ds = pd.read_csv(ds_path, sep=',', header=0)
    create_save_train_val_ds(
        ds,
        0.3,
        train_path,
        val_path,
        False,
        False)
