from ml_sdk.dataset.model_feeder.ds_creator_builder import DsCreatorBuilder
from ml_sdk.dataset.model_feeder.mlp_ds_creator import MLPDsCreator


class MLPDsCreatorBuilder(DsCreatorBuilder):
    def __init__(self, train_ds_path, val_ds_path, file_record_struct, batch_size, output_nbr):
        super(MLPDsCreatorBuilder, self).__init__(train_ds_path, val_ds_path, file_record_struct, batch_size)
        self._output_nbr = output_nbr

    def get_ds_creator(self, repeat) -> MLPDsCreator:
        return MLPDsCreator(
            self._train_ds_path,
            self._val_ds_path,
            self._file_record_struct,
            repeat,
            self._batch_size,
            False,
            self._output_nbr)
