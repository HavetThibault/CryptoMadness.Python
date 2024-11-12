from typing import Optional

from ml_sdk.optimization.parameters_matrix_generator import params_set_to_str
from ml_sdk.training.parameter import Parameter
from ml_sdk.training.training_epoch_stats import TrainingEpochStats


class TrainingResults:
    def __init__(self, params_set, stats):
        self._params_set: list[Parameter] = params_set
        self._stats: Optional[list[TrainingEpochStats]] = stats

    def get_params_set(self):
        return self._params_set

    def get_stats(self):
        return self._stats

    def get_stats_count(self):
        return len(self._stats)

    def add_stat(self, stat: TrainingEpochStats):
        self._stats.append(stat)

    def set_none(self):
        self._stats = None

    def __str__(self):
        result_str = f'params_set: {params_set_to_str(self._params_set)}; ['
        for stat in self._stats:
            result_str += str(stat)
        return result_str
