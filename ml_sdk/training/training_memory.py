import math
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from ml_sdk.training.error_calculator import ErrorCalculator
from ml_sdk.training.model_filename_mgmt import WEIGHTS_FILE_EXTENSION, MODEL_FILE_EXTENSION
from ml_sdk.training.training_epoch_stats import TrainingEpochStats


class TrainingMemory:
    DEFAULT_TARGET_COL = 'target'
    FILE_EXT = '.bin'

    @staticmethod
    def load_instance(memory_filepath):
        with open(memory_filepath, 'rb') as file:
            training_memory = pickle.load(file)
            return training_memory

    @staticmethod
    def get_instance(memory_filepath, params_matrix, weight_dir, repeat):
        if os.path.isfile(memory_filepath):
            training_mem = TrainingMemory.load_instance(memory_filepath)
            for i, params_set in training_mem.get_params_matrix().items():
                for j, param in enumerate(params_set):
                    assert math.isclose(param, params_matrix[i][j])
            training_mem._weight_dir = weight_dir
            training_mem._memory_file_path = memory_filepath
            return training_mem
        return TrainingMemory(memory_filepath, params_matrix, weight_dir, repeat)

    def __init__(self, memory_filepath, params_matrix: list[list], weight_dir, repeat):
        self._memory_file_path = memory_filepath
        self._params_matrix: dict[int, list] = self._to_index_dict(params_matrix)
        self._weight_dir = weight_dir
        self.repeat = repeat
        self._trainings_stats: dict[int, Optional[dict[int, TrainingEpochStats]]] = {}
        self._current_params_set_index = 0
        self._current_params_set_repeat_index = 0

    def get_repeat(self):
        return self.repeat

    @staticmethod
    def _to_index_dict(params_matrix: list[list]) -> dict[int, list]:
        index_dict = {}
        for i in range(len(params_matrix)):
            index_dict[i] = params_matrix[i]
        return index_dict

    def get_all_training_stats(self):
        return self._trainings_stats

    def get_training_stats(self, index, sub_index) -> Optional[TrainingEpochStats]:
        if self._trainings_stats[index] is None:
            return None
        return self._trainings_stats[index][sub_index]

    def get_target_df(self, error_calc: ErrorCalculator, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
        target = []
        for i in range(len(self._params_matrix)):
            stats = self._trainings_stats[i]
            if stats is not None:
                errors = []
                for _, stat in stats.items():
                    errors.append(error_calc.get_error(float(stat.loss), float(stat.val_loss)))
                if self.repeat > 1:
                    target.append(np.mean(errors))
                else:
                    target.append(errors[0])
            else:
                target.append(None)
        return pd.DataFrame(target, columns=[target_col], dtype='Float64')

    def get_all_target_df(self, error_calc: ErrorCalculator, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
        target = []
        index = []
        for i in range(len(self._params_matrix)):
            stats = self._trainings_stats[i]
            if stats is not None:
                for j, (_, stat) in enumerate(stats.items()):
                    target.append(error_calc.get_error(stat.loss, stat.val_loss))
                    index.append((i, j))
            else:
                for j in range(self.repeat):
                    target.append(None)
                    index.append((i, j))
        return pd.DataFrame(target, columns=[target_col], dtype='Float64', index=pd.MultiIndex.from_tuples(index))

    def get_params_df(self, cols: list[str]) -> pd.DataFrame:
        df_cols = []
        for i in range(len(self._params_matrix)):
            df_cols.append(self._params_matrix[i])
        return pd.DataFrame(df_cols, columns=cols, index=[i for i in range(len(self._params_matrix))])

    def get_all_params_df(self, cols: list[str]) -> pd.DataFrame:
        params_df = []
        index = []
        for i in range(len(self._params_matrix)):
            params_set = self._params_matrix[i]
            for j in range(self.repeat):
                params_df.append(params_set)
                index.append((i, j))
        return pd.DataFrame(params_df, columns=cols, index=pd.MultiIndex.from_tuples(index))

    def get_params_set_cnt(self):
        return len(self._params_matrix)

    def has_next_params_set(self) -> bool:
        return self._current_params_set_index < len(self._params_matrix)

    def _next_training_stats(self):
        self._current_params_set_repeat_index += 1
        if self._current_params_set_repeat_index == self.repeat:
            self._next_params_set()

    def _next_params_set(self):
        self._current_params_set_index += 1
        self._current_params_set_repeat_index = 0

    def get_current_params_set(self) -> list:
        return self._params_matrix[self._current_params_set_index]

    def get_current_trainings_index(self) -> tuple[int, int]:
        return self._current_params_set_index, self._current_params_set_repeat_index

    def add_training_stats(self, training_stats: Optional[TrainingEpochStats]):
        if training_stats is not None:
            if self._current_params_set_index not in self._trainings_stats:
                self._trainings_stats[self._current_params_set_index] = \
                    {self._current_params_set_repeat_index: training_stats}
            else:
                self._trainings_stats[self._current_params_set_index][self._current_params_set_repeat_index] = (
                    training_stats)
            self._next_training_stats()
        else:
            if self._current_params_set_index in self._trainings_stats:
                raise Exception('Weird behavior: the parameter "training_stats" is None, but the previous training '
                                'stats for this parameters set was valid !')
            self._trainings_stats[self._current_params_set_index] = None
            self._next_params_set()

    def get_params_matrix(self):
        return self._params_matrix

    def get_best_training_stats_index(self, error_calc: ErrorCalculator) -> tuple[int, int]:
        min_error = None
        min_error_index = None
        for index, stats in self._trainings_stats.items():
            if stats is None:
                continue
            for sub_index, stat in stats.items():
                error = error_calc.get_error(stat.loss, stat.val_loss)
                if min_error is None or error < min_error:
                    min_error_index = (index, sub_index)
                    min_error = error
        return min_error_index

    def get_mean_best_params_set_index(self, error_calc: ErrorCalculator) -> int:
        min_error = None
        min_error_index = None
        for index, stats in self._trainings_stats.items():
            if stats is None:
                continue
            errors = []
            for sub_index, stat in stats.items():
                errors.append(error_calc.get_error(float(stat.loss), float(stat.val_loss)))
            if self.repeat > 1:
                error = np.mean(errors)
            else:
                error = errors[0]
            if min_error is None or error < min_error:
                min_error_index = index
                min_error = error
        return min_error_index

    def get_current_model_filename(self, model_name, weights_only) -> str:
        return self.get_model_filename(
            model_name,
            self._current_params_set_index,
            self._current_params_set_repeat_index,
            weights_only)

    def get_all_archived_files(self, model_name, weights_only) -> list[str]:
        files = []
        for index, value in self._trainings_stats.items():
            if value is None:
                continue
            for sub_index, stats in value.items():
                files.append(self.get_model_filename(model_name, index, sub_index, weights_only))
        return files

    @staticmethod
    def get_model_filename(model_name, index, sub_index, weights_only) -> str:
        filename = f'{model_name}__{index}__{sub_index}'
        if weights_only:
            return filename + WEIGHTS_FILE_EXTENSION
        return filename + MODEL_FILE_EXTENSION

    @staticmethod
    def get_filename_index_sub_index(filename: str) -> tuple[int, int]:
        dot_index = filename.index('.')
        no_ext_filename = filename[:dot_index]
        print(no_ext_filename)
        split = no_ext_filename.split('__')
        return int(split[1]), int(split[2])

    @staticmethod
    def get_name_and_index(model_filename: str) -> tuple[str, tuple[int, int]]:
        if model_filename.endswith(WEIGHTS_FILE_EXTENSION):
            no_ext_model_file_name = model_filename[:-len(WEIGHTS_FILE_EXTENSION)]
        else:
            no_ext_model_file_name = model_filename[:-len(MODEL_FILE_EXTENSION)]
        filename_split = no_ext_model_file_name.split('__')
        return filename_split[0], (int(filename_split[1]), int(filename_split[2]))

    def get_best_params_set(self, error_calc: ErrorCalculator) -> list[float]:
        best_params_set_index = self.get_best_training_stats_index(error_calc)
        return self._params_matrix[best_params_set_index[0]]

    def get_best_mean_params_set(self, error_calc: ErrorCalculator) -> list[float]:
        best_params_set_index = self.get_mean_best_params_set_index(error_calc)
        return self._params_matrix[best_params_set_index]

    def save(self):
        with open(self._memory_file_path, 'wb') as file:
            return pickle.dump(self, file)

    def __str__(self):
        memory_str = 'params_set_cnt: ' + str(self.get_params_set_cnt())
        memory_str += ' - params_matrix: ' + str(self._params_matrix)
        memory_str += ' - trainings_stats: ['
        for i, stat in enumerate(self._trainings_stats):
            if i < len(self._trainings_stats) - 1:
                memory_str += str(stat) + ', '
            else:
                memory_str += str(stat)
        return memory_str + ']'

