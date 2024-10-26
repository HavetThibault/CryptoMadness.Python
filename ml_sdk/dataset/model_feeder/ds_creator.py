import tensorflow as tf
import pandas as pd


class DsCreator:
    def __init__(self, train_csv_path: str, val_csv_path: str, file_record_struct, repeat_ds: int, batch_size: int,
                 use_cache):
        self._train_csv_path = train_csv_path
        self._val_csv_path = val_csv_path
        self._file_record_struct = file_record_struct
        self._repeat_ds = repeat_ds
        self._use_cache = use_cache
        self._batch_size = batch_size

    def _parse_label_file(self, line_record) -> tuple:
        raise NotImplementedError()

    def _configure_for_performance(self, ds, ds_len, batch_size, shuffle):
        if self._use_cache:
            ds = ds.cache()

        if shuffle:
            ds = ds.shuffle(buffer_size=ds_len)

        ds = ds.repeat(self._repeat_ds)

        ds = ds.batch(batch_size)
        # Load next rows while current row is being processed
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def _load_dataset(self, ds_path: str, verbose=False) -> tf.data.Dataset:
        line_ds = tf.data.TextLineDataset(ds_path)
        if verbose:
            for element in line_ds.as_numpy_iterator():
                print(element)
        # skip headers
        line_ds = line_ds.skip(1)
        img_tuple_ds = line_ds.map(self._parse_label_file, num_parallel_calls=tf.data.AUTOTUNE)
        return img_tuple_ds

    def get_train_ds(self, shuffle=False, verbose=False) -> tuple[tf.data.Dataset, int]:
        train_set_len = len(self.get_pandas_val_df())
        if verbose:
            print('Train set len :', str(train_set_len))
        train_ds = self._load_dataset(self._train_csv_path, verbose)
        perf_train_ds = self._configure_for_performance(train_ds, train_set_len, self._batch_size, shuffle)
        return perf_train_ds, train_set_len

    def get_val_ds(self, shuffle=False, verbose=False, batch_one=False) -> tuple[tf.data.Dataset, int]:
        val_set_len = len(self.get_pandas_val_df())
        if verbose:
            print('Val set :', str(val_set_len))
        val_ds = self._load_dataset(self._val_csv_path, verbose)
        if batch_one:
            batch_size = 1
        else:
            batch_size = self._batch_size
        perf_val_ds = self._configure_for_performance(val_ds, val_set_len, batch_size, shuffle)
        return perf_val_ds, val_set_len

    def get_pandas_train_df(self):
        return self._get_pandas_df(self._train_csv_path)

    def get_pandas_val_df(self):
        return self._get_pandas_df(self._val_csv_path)

    @staticmethod
    def _get_pandas_df(path):
        return pd.read_csv(path, header=0)
