from ml_sdk.dataset.model_feeder.ds_creator import DsCreator
import tensorflow as tf

from ml_sdk.model.creator.mlp_model_creator import MLPModelCreator


class MLPDsCreator(DsCreator):
    def __init__(self, train_csv_path: str, val_csv_path: str, file_record_struct, repeat_ds: int, batch_size: int,
                 use_cache, output_nbr):
        super(MLPDsCreator, self).__init__(
            train_csv_path,
            val_csv_path,
            file_record_struct,
            repeat_ds,
            batch_size,
            use_cache)
        self._output_nbr = output_nbr
        self._input_nbr = len(self._file_record_struct) - output_nbr

    def _parse_label_file(self, line_record) -> tuple:
        fields = tf.io.decode_csv(line_record, self._file_record_struct)
        input_dict = {MLPModelCreator.INPUT_NAME: fields[:self._input_nbr]}
        output_dict = {MLPModelCreator.OUTPUT_NAME: fields[self._input_nbr:]}
        return input_dict, output_dict
