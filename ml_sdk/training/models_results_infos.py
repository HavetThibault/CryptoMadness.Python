import os
import pickle
from typing import Optional

from ml_sdk.training.best_model_checkpoint import BestModelCheckpoint
from ml_sdk.training.model_filename_mgmt import WEIGHTS_FILE_EXTENSION, MODEL_FILE_EXTENSION
from ml_sdk.training.parameter import Parameter
from ml_sdk.training.training_epoch_stats import TrainingEpochStats
from ml_sdk.training.training_results import TrainingResults


class ModelsResultsInfos:
    FILE_EXT = '.bin'
    ARCHIVED_DIR = 'Archived/'
    WEIGHTS_DIR = 'Temp/'
    METRICS_FILE_END = 'metrics.csv'
    PREDICTIONS_DIR = 'Predictions/'

    @staticmethod
    def load_instance(memory_filepath):
        with open(memory_filepath, 'rb') as file:
            training_memory = pickle.load(file)
            return training_memory

    @staticmethod
    def get_instance(memory_filepath, dest_dir):
        if os.path.isfile(memory_filepath):
            training_mem = ModelsResultsInfos.load_instance(memory_filepath)
            training_mem._dest_dir = dest_dir
            training_mem._memory_file_path = memory_filepath
            return training_mem
        return ModelsResultsInfos(memory_filepath, dest_dir)

    @staticmethod
    def static_get_metrics_path(dest_dir, model_name):
        return dest_dir + model_name + '__' + ModelsResultsInfos.METRICS_FILE_END
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

    def __init__(self, memory_filepath, dest_dir):
        self._memory_file_path = memory_filepath
        self._dest_dir = dest_dir
        self._trainings_results: list[TrainingResults] = []

    def get_params_set_index(self, params_set: list[Parameter]) -> Optional[int]:
        for i, result in enumerate(self._trainings_results):
            if result.get_params_set() == params_set:
                return i
        return len(self._trainings_results)

    def get_dest_dir(self):
        return self._dest_dir

    def get_weights_dir(self):
        return self._dest_dir + self.WEIGHTS_DIR

    def get_archived_dir(self):
        return self._dest_dir + self.ARCHIVED_DIR

    def get_metrics_path(self, model_name):
        return self.static_get_metrics_path(self._dest_dir, model_name)

    def get_trainings_results(self):
        return self._trainings_results

    def get_params_set_cnt(self):
        cnt = 0
        for result in self._trainings_results:
            cnt += result.get_stats_count()
        return cnt

    def get_training_results(self, params_set: list[Parameter]):
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

def archive(models_results_infos: ModelsResultsInfos, params_set, model_creator, serializer: BestModelCheckpoint,
            training_hist, sub_params_set_index, weights_only):
    if training_hist is not None:
        params_set_index = models_results_infos.get_params_set_index(params_set)
        best_model_path = models_results_infos.get_weights_dir() + serializer.get_best_model_filename()
        os.rename(
            best_model_path,
            models_results_infos.get_archived_dir() + models_results_infos.get_model_filename(
                model_creator.get_model_name(), params_set_index, sub_params_set_index, weights_only))
        models_results_infos.add_training_stats(params_set, serializer.get_best_stats())
    else:
        models_results_infos.add_training_stats(params_set, None)
    models_results_infos.save()