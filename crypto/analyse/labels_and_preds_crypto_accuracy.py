import pandas as pd

from ml_sdk.analyze.labels_and_preds.labels_and_preds_processor import LabelsAndPredsProcessor
from ml_sdk.analyze.predictions_metrics import PREDICTION_COL_END
from ml_sdk.analyze.success_rate import SuccessRate
from ml_sdk.dataset.cook.ds_source_file import read_dataset, write_dataset
from ml_sdk.model.creator.mlp_model_creator import MLPModelCreator
from ml_sdk.optimization.parameters_matrix_generator import params_set_to_str


class LabelsAndPredsCryptoAccuracy(LabelsAndPredsProcessor):
    COLUMNS = ['params_set', 'increase_success', 'increase_total', 'increase_success_rate',
               'decrease_success', 'decrease_total', 'decrease_success_rate']

    def __init__(self, val_filepath, metrics_path, min_close, max_close):
        self._val_filepath: str = val_filepath
        self._metrics_path = metrics_path
        self._min_close = min_close
        self._max_close = max_close
        self._metrics_rows = []

    def process_start(self):
        pass

    def process_end(self):
        write_dataset(
            pd.DataFrame(self._metrics_rows, columns=self.COLUMNS),
            self._metrics_path,
            header=True)

    def process(self, params_set, filename, predictions: pd.DataFrame):
        val_df = read_dataset(self._val_filepath, header=0)
        predictions_iter = iter(predictions.iloc)
        val_iter = iter(val_df.iloc)
        increase_success = SuccessRate()
        decrease_success = SuccessRate()
        last_close_col = self.get_last_close_col(list(val_df.columns))
        for predictions_row in predictions_iter:
            val_row = next(val_iter)
            prediction = predictions_row[MLPModelCreator.OUTPUT_NAME + PREDICTION_COL_END]
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
        self._metrics_rows.append([
            params_set_to_str(params_set),
            increase_success.get_right(),
            increase_success.get_total(),
            f'{increase_success.get_rate() * 100: 0.3f}%',
            decrease_success.get_right(),
            decrease_success.get_total(),
            f'{decrease_success.get_rate() * 100: 0.3f}%'])

    @staticmethod
    def get_last_close_col(cols: list[str]) -> str:
        max_index = None
        last_close = None
        for col in cols:
            if col.startswith('Close'):
                index = int(col.split('_')[1])
                if max_index is None or index > max_index:
                    max_index = index
                    last_close = col
        return last_close
