import math
import os

import pandas as pd

from helper_sdk.work_progress_state import WorkProgressState
from ml_sdk.model.creator.model_creator import ModelCreator
from ml_sdk.optimization.parameters_matrix_generator import params_set_to_str
from ml_sdk.optimization.regression.ds_transformer import DSTransformer
from ml_sdk.optimization.regression.reg_calculator import RegCalculator
from ml_sdk.optimization.regression.reg_frame_finder import CONST_COL
from ml_sdk.plot.model_stats import plot_metrics
from ml_sdk.training.best_model_checkpoint import BestModelCheckpoint
from ml_sdk.training.model_filename_mgmt import WEIGHTS_FILE_EXTENSION, MODEL_FILE_EXTENSION
from ml_sdk.training.models_results_infos import ModelsResultsInfos, archive


def run_trainings(
        params_matrix,
        model_creator: ModelCreator,
        memory_filepath: str,
        dest_dir: str,
        iterations_nbr: int,
        error_calc,
        repeat,
        additional_callbacks=None,
        plot_errors=False,
        ylim=None,
        verbose=0,
        optimizer=None,
        weights_only=True,
        give_up_cpu=True,
        loss=None,
        work_progress: WorkProgressState=None,
        on_cpu=False) -> ModelsResultsInfos:

    training_memory = ModelsResultsInfos.get_instance(memory_filepath, dest_dir)
    print('Created/loaded training memory: ', str(training_memory))
    weight_dir = training_memory.get_weights_dir()
    archived_dir = training_memory.get_archived_dir()

    if not os.path.exists(archived_dir):
        os.mkdir(archived_dir)
    if os.path.exists(training_memory.get_weights_dir()):
        clean_weights_dir(weight_dir, weights_only)

    serializer = BestModelCheckpoint(weight_dir, model_creator.get_model_name(), error_calc, verbose,
                                     weights_only=weights_only)
    if additional_callbacks is None:
        additional_callbacks = []
    k = 0
    for params_set in params_matrix:
        if training_memory.get_training_results(params_set) is None:
            break
        k += 1
    for params_set_index in range(k, len(params_matrix)):
        params_set = params_matrix[params_set_index]
        current_training_results = training_memory.get_training_results(params_set)
        while (current_training_results is None
               or
               current_training_results.get_stats() is None
               or
               current_training_results.get_stats_count() >= repeat):
            if current_training_results is not None:
                sub_params_set_index = current_training_results.get_stats_count()
            else:
                sub_params_set_index = 0
            print(f'{params_set_index}.{sub_params_set_index}. Training for: {params_set_to_str(params_set)}')
            training_hist = model_creator.create_and_train_model(
                params_set,
                iterations_nbr,
                [serializer] + additional_callbacks,
                give_up_cpu,
                optimizer=optimizer,
                verbose=verbose,
                loss=loss,
                on_cpu=on_cpu)

            if training_hist is not None:
                if plot_errors:
                    plot_metrics(
                        training_hist.history,
                        model_creator.get_model_name() + params_set_to_str(params_set),
                        ['loss', 'val_loss'],
                        ylim=ylim)

            archive(training_memory, params_set, model_creator, serializer, training_hist, sub_params_set_index,
                    weights_only)

            if work_progress is not None:
                work_progress.increment_done()

            if current_training_results is None:
                current_training_results = training_memory.get_training_results(params_set)
    return training_memory


def clean_weights_dir(weights_dir: str, weights_only: bool):
    for filename in os.listdir(weights_dir):
        if (filename.endswith(WEIGHTS_FILE_EXTENSION) and weights_only or
                filename.endswith(MODEL_FILE_EXTENSION) and not weights_only):
            print('Removing ' + weights_dir + filename)
            os.remove(weights_dir + filename)


def lowest_error_params_set(best_frame, matrix_params, columns: list[str], ds_transformer: DSTransformer,
                            reg_calculator: RegCalculator) -> tuple[pd.DataFrame, float]:
    all_params_matrix = pd.DataFrame(matrix_params, columns=columns)
    transformed_params_matrix = ds_transformer.transform(all_params_matrix)
    best_params_set = None
    lowest_error = None
    print(transformed_params_matrix[best_frame])
    for params_set in transformed_params_matrix[best_frame].itertuples():
        no_index_params_set = params_set[1:]
        all_params_zero = True
        for i, param in enumerate(no_index_params_set):
            if best_frame[i] != CONST_COL and not math.isclose(param, 0, rel_tol=0.001):
                all_params_zero = False
                break
        if not all_params_zero:
            error = reg_calculator.predict(no_index_params_set)
            if lowest_error is None or error < lowest_error:
                lowest_error = error
                best_params_set = params_set
    return best_params_set, lowest_error
