import pandas as pd

from helper_sdk.work_progress_state import WorkProgressState
from ml_sdk.analyse.class_model_labels_and_preds import ClassModelLabelsAndPreds
from ml_sdk.analyse.model_labels_and_preds import ModelLabelsAndPreds
from ml_sdk.analyse.model_labels_and_preds_instantiator import ModelLabelsAndPredsInstantiator
from ml_sdk.analyse.model_output_error import ModelsOutputsErrors
from ml_sdk.analyse.predictions_metrics import get_params_intervals
from ml_sdk.model.creator.model_creator import ModelCreator
from ml_sdk.plot.model_stats import trainings_errors_boxplot
from ml_sdk.training.trainings_memory import TrainingsMemory


def calculate_models_val_labels_and_preds(training_memories_model_creator: list[ModelCreator],
                                          training_mem_dir,
                                          model_labels_and_preds_inst: ModelLabelsAndPredsInstantiator,
                                          progress: WorkProgressState) \
        -> list[ModelLabelsAndPreds]:
    models_labels_and_preds = []
    for model_creator in training_memories_model_creator:
        print(f'Calculating predictions of {model_creator.get_model_name()}...')
        training_mem_name = model_creator.get_model_name()
        training_mem: TrainingsMemory = TrainingsMemory.load_instance(
            training_mem_dir + training_mem_name + TrainingsMemory.FILE_EXT)
        trainings_results = training_mem.get_trainings_results()
        for i, training_results in enumerate(trainings_results):
            if training_results.get_stats() is None:
                continue
            params_set = training_results.get_params_set()
            for k in range(training_results.get_stats_count()):
                weights_file = training_mem.get_model_filename(model_creator.get_model_name(), i, k, True)
                model = model_creator.load_model_from_weights(
                    params_set, training_mem_dir + 'archived/' + weights_file)
                val_ds, val_len = model_creator.get_ds_creator_builder().get_ds_creator(1).get_val_ds()
                models_labels_and_preds.append(
                    model_labels_and_preds_inst.instantiate(
                        model,
                        val_ds,
                        val_len,
                        model_creator.get_batch_size(),
                        model_creator.get_model_name(),
                        params_set,
                        progress,
                        i,
                        k))
    return models_labels_and_preds


def get_best_reg_models_perfs_df(models_labels_and_preds: list[ModelLabelsAndPreds], output_param_names,
                                 preds_error_calc, per_param_preds_error_calc) -> tuple[pd.DataFrame, dict[str, ModelLabelsAndPreds]]:
    best_params_set_errors: dict[str, tuple[ModelLabelsAndPreds, float]] = {}
    for model_labels_and_preds in models_labels_and_preds:
        model_name = model_labels_and_preds.get_model_name()
        labels_and_preds = model_labels_and_preds.get_labels_and_preds()

        error = preds_error_calc(labels_and_preds)  # get_errors_mean_square
        if model_name not in best_params_set_errors or best_params_set_errors[model_name][1] > error:
            best_params_set_errors[model_name] = (model_labels_and_preds, error)

    colnames = ['Name', 'Params set', 'Error mean']
    for param_name in output_param_names:
        colnames.append(param_name + ' mean')
        colnames.append(param_name + ' std')

    result_df_rows = []
    best_models = {}
    for model_name, (model_labels_and_preds, error) in best_params_set_errors.items():
        labels_and_preds = model_labels_and_preds.get_labels_and_preds()
        params_set = model_labels_and_preds.get_param_set()
        best_models[model_name] = model_labels_and_preds

        errors_mean_std = per_param_preds_error_calc(labels_and_preds)  # get_per_param_abs_errors_mean_std
        row = [model_name, str(params_set), f'{error:0.2f}']
        for i, (errors_mean, errors_std) in enumerate(errors_mean_std):
            row.append(errors_mean)
            row.append(errors_std)
        result_df_rows.append(row)
    return pd.DataFrame(result_df_rows, columns=colnames), best_models


def get_models_errors_df(models_labels_and_preds: list[ModelLabelsAndPreds], output_param_names,
                         get_per_param_error) -> list[ModelsOutputsErrors]:
    models_errors = []
    for output in output_param_names:
        models_errors.append(ModelsOutputsErrors(output))

    for model_preds in models_labels_and_preds:
        labels_and_preds = model_preds.get_labels_and_preds()
        errors_mean_std = get_per_param_error(labels_and_preds)
        for i, (errors_mean, errors_std) in enumerate(errors_mean_std):
            models_errors[i].add_model_error(model_preds.get_model_name(), model_preds.get_param_set(), errors_mean, errors_std)
    return models_errors


def get_best_class_models_perfs_df(models_labels_and_preds: list[ClassModelLabelsAndPreds], output_param_names,
                                   preds_error_calc, per_param_preds_error_calc) -> pd.DataFrame:
    best_params_set_errors: dict[str, tuple[ClassModelLabelsAndPreds, float]] = {}
    for model_labels_and_preds in models_labels_and_preds:
        model_name = model_labels_and_preds.get_model_name()
        labels_and_preds = model_labels_and_preds.get_labels_and_preds()

        error = preds_error_calc(labels_and_preds)  # get_errors_mean_square
        if model_name not in best_params_set_errors or best_params_set_errors[model_name][1] > error:
            best_params_set_errors[model_name] = (model_labels_and_preds, error)

    colnames = ['Name', 'Params set', 'Error mean']
    for param_name in output_param_names:
        colnames.append(param_name + ' mean')
        colnames.append(param_name + ' std')
        colnames.append(param_name + ' accuracy')

    result_df_rows = []
    for model_name, (model_labels_and_preds, error) in best_params_set_errors.items():
        labels_and_preds = model_labels_and_preds.get_labels_and_preds()
        params_set = model_labels_and_preds.get_param_set()

        errors_mean_std = per_param_preds_error_calc(labels_and_preds)  # get_per_param_abs_errors_mean_std
        row = [model_name, str(params_set), f'{error:0.2f}']
        for i, (errors_mean, errors_std) in enumerate(errors_mean_std):
            row.append(errors_mean)
            row.append(errors_std)
            row.append(model_labels_and_preds.get_params_accuracy()[i])
        result_df_rows.append(row)

    return pd.DataFrame(result_df_rows, columns=colnames)


def show_models_errors_boxplot(models_labels_and_preds: list[ModelLabelsAndPreds], preds_error_calc):
    model_name_errors = {}
    for model_labels_and_preds in models_labels_and_preds:
        error = preds_error_calc(model_labels_and_preds.get_labels_and_preds())  # get_errors_mean_square
        model_name = model_labels_and_preds.get_model_name()
        if model_name not in model_name_errors:
            model_name_errors[model_name] = [error]
        else:
            model_name_errors[model_name].append(error)

    sol_cols = list(model_name_errors.keys())

    errors_list = []
    for model_name, errors in model_name_errors.items():
        errors_list.append(pd.DataFrame(errors))

    trainings_errors_boxplot(errors_list, (8, 5), sol_cols)


def get_param_intervals_df(ds, output_param_names) -> pd.DataFrame:
    intervals_rows = [[], []]
    params_intervals = get_params_intervals(ds[output_param_names])
    for min_val, max_val in params_intervals:
        intervals_rows[0].append(min_val)
        intervals_rows[1].append(max_val)
    return pd.DataFrame(intervals_rows, columns=output_param_names)
