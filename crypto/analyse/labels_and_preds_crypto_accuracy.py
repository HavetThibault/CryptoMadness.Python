import pandas as pd

from helper_sdk.file_helper import rm_file_ext
from ml_sdk.analyse.labels_and_preds.labels_and_preds_processor import LabelsAndPredsProcessor
from ml_sdk.analyse.success_rate import SuccessRate
from ml_sdk.dataset.cook.ds_source_file import read_dataset, write_dataset
from ml_sdk.model.creator.mlp_model_creator import MLPModelCreator


class LabelsAndPredsCryptoAccuracy(LabelsAndPredsProcessor):
    def __init__(self, val_filepath, dest_folder, min_close, max_close):
        self._val_filepath: str = val_filepath
        self._dest_folder = dest_folder
        self._min_close = min_close
        self._max_close = max_close

    def process(self, filename, predictions: pd.DataFrame):
        val_df = read_dataset(self._val_filepath, header=0)
        predictions_iter = iter(predictions.iloc)
        val_iter = iter(val_df.iloc)
        increase_success = SuccessRate()
        decrease_success = SuccessRate()
        last_close_col = self._get_last_close_col(list(val_df.columns))
        for predictions_row in predictions_iter:
            val_row = next(val_iter)
            prediction = predictions_row[MLPModelCreator.OUTPUT_NAME + ' prediction']
            label = predictions_row[MLPModelCreator.OUTPUT_NAME]
            last_close = val_row[last_close_col] * (self._max_close - self._min_close) + self._min_close
            if label - last_close > 0:
                if prediction - last_close > 0:
                    increase_success.add_right()
                else:
                    increase_success.add_wrong()
            else:
                if prediction - last_close < 0:
                    decrease_success.add_right()
                else:
                    decrease_success.add_wrong()
        write_dataset(
            pd.DataFrame(
                [[increase_success.get_rate(), decrease_success.get_rate()]],
                columns=['Increase success rate', 'Decrease success rate']),
            self._dest_folder + rm_file_ext(filename) + 'metrics.csv'
        )

    @staticmethod
    def _get_last_close_col(cols: list[str]) -> str:
        max_index = None
        last_close = None
        for col in cols:
            if col.startswith('Close'):
                index = int(col.split('_')[1])
                if max_index is None or index > max_index:
                    max_index = index
                    last_close = col
        return last_close


