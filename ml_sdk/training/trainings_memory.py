import math
import os
import pickle
from typing import Optional

from ml_sdk.training.model_filename_mgmt import WEIGHTS_FILE_EXTENSION, MODEL_FILE_EXTENSION
from ml_sdk.training.training_epoch_stats import TrainingEpochStats
from ml_sdk.training.training_results import TrainingResults


class TrainingsMemory:
    DEFAULT_TARGET_COL = 'target'
    FILE_EXT = '.bin'

    @staticmethod
    def load_instance(memory_filepath):
        with open(memory_filepath, 'rb') as file:
            training_memory = pickle.load(file)
            return training_memory

    @staticmethod
    def get_instance(memory_filepath, weight_dir):
        if os.path.isfile(memory_filepath):
            training_mem = TrainingsMemory.load_instance(memory_filepath)
            training_mem._weight_dir = weight_dir
            training_mem._memory_file_path = memory_filepath
            return training_mem
        return TrainingsMemory(memory_filepath, weight_dir)

    def __init__(self, memory_filepath, weight_dir):
        self._memory_file_path = memory_filepath
        self._weight_dir = weight_dir
        self._trainings_results: list[TrainingResults] = []

    def get_trainings_results(self):
        return self._trainings_results

    def get_params_set_cnt(self):
        cnt = 0
        for result in self._trainings_results:
            cnt += result.get_stats_count()
        return cnt

    def get_training_results(self, params_set):
        for result in self._trainings_results:
            if result.get_params_set() == params_set:
                return result
        return None

    def add_training_stats(self, params_set, stat: Optional[TrainingEpochStats]):
        for result in self._trainings_results:
            if result.get_params_set() == params_set:
                if stat is not None:
                    result.add_stat(stat)
                else:
                    result.set_none()
                return
        self._trainings_results.append(TrainingResults(params_set, [stat]))

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

    def save(self):
        with open(self._memory_file_path, 'wb') as file:
            return pickle.dump(self, file)

    def __str__(self):
        memory_str = 'params_set_cnt: ' + str(self.get_params_set_cnt())
        memory_str += ' - trainings_results: ['
        for i, result in enumerate(self._trainings_results):
            if i < len(self._trainings_results) - 1:
                memory_str += str(result) + ', '
            else:
                memory_str += str(result)
        return memory_str + ']'

