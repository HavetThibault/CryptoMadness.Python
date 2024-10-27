import tensorflow as tf

from ml_sdk.dataset.file.file_record_struct import FileRecordStruct
from ml_sdk.dataset.model_feeder.ds_creator import DsCreator


class DsCreatorBuilder:
    def __init__(self, train_ds_path, val_ds_path, file_record_struct, batch_size):
        self._train_ds_path: str = train_ds_path
        self._val_ds_path: str = val_ds_path
        self._file_record_struct: FileRecordStruct = file_record_struct
        self._batch_size: int = batch_size

    def get_train_path(self):
        return self._train_ds_path

    def get_val_path(self):
        return self._val_ds_path

    def get_ds_creator(self, iteration_nbr) -> DsCreator:
        raise NotImplementedError('Not implemented get_dataset_creator method')

    def get_train_val_ds(self, iteration_nbr) -> tuple[tuple[tf.data.Dataset, int], tuple[tf.data.Dataset, int]]:
        dataset_creator = self.get_ds_creator(iteration_nbr)
        train = dataset_creator.get_train_ds()
        val = dataset_creator.get_val_ds()
        return train, val
