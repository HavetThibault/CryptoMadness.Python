import pandas as pd

from ml_sdk.training.training_epoch_stats import TrainingEpochStats
from ml_sdk.training.training_results import TrainingResults


class LabelsAndPredsProcessor:

    def process_start(self):
        raise NotImplementedError()

    def process_end(self):
        raise NotImplementedError()

    def process(self, params_set, stat: TrainingEpochStats, filename, predictions: pd.DataFrame):
        raise NotImplementedError()
