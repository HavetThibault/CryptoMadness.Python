import os

from helper_sdk.file_helper import rm_file_ext
from helper_sdk.work_progress_state import WorkProgressState
from ml_sdk.analyze.labels_and_preds.labels_and_preds_processor import LabelsAndPredsProcessor
from ml_sdk.analyze.model_labels_and_preds_instantiator import ModelLabelsAndPredsInstantiator
from ml_sdk.analyze.trainings_analyses import calculate_models_val_labels_and_preds, \
    calculate_save_models_val_labels_and_preds
from ml_sdk.dataset.cook.ds_source_file import write_dataset, read_dataset
from ml_sdk.model.creator.model_creator import ModelCreator
from ml_sdk.training.models_results_infos import ModelsResultsInfos


def analyze(model_creator: ModelCreator, labels_and_preds_instatiator: ModelLabelsAndPredsInstantiator, files_dir: str,
            progress: WorkProgressState, progress_ratio=0.01):
    ds_creator = model_creator.get_ds_creator_builder().get_ds_creator(1)

    val_ds, val_len = ds_creator.get_val_ds()
    training_mem_name = model_creator.get_model_name()
    results: ModelsResultsInfos = ModelsResultsInfos.load_instance(
        files_dir + training_mem_name + ModelsResultsInfos.FILE_EXT)
    progress.reset(
        val_len * results.get_params_set_cnt(),
        int(val_len * results.get_params_set_cnt() * progress_ratio))
    progress.start_resume()

    pred_dir = results.get_predictions_dir()
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    calculate_save_models_val_labels_and_preds(
        model_creator,
        files_dir,
        ds_creator.get_pandas_val_df(),
        labels_and_preds_instatiator,
        progress)


def apply_labels_and_preds(model_creator: ModelCreator, files_dir: str, progress: WorkProgressState,
                           apply: LabelsAndPredsProcessor):
    training_mem_name = model_creator.get_model_name()
    training_mem: ModelsResultsInfos = ModelsResultsInfos.load_instance(
        files_dir + training_mem_name + ModelsResultsInfos.FILE_EXT)
    trainings_results = training_mem.get_trainings_results()
    progress.reset(training_mem.get_params_set_cnt(), 1)
    progress.start_resume()
    apply.process_start()
    for i, training_results in enumerate(trainings_results):
        if training_results.get_stats() is None:
            continue
        for k, stat in enumerate(training_results.get_stats()):
            predictions_filename = ModelsResultsInfos.get_model_filename(
                model_creator.get_model_name(), i, k, False)
            predictions_filename = rm_file_ext(predictions_filename) + '.csv'
            predictions_df = read_dataset(files_dir + ModelsResultsInfos.PREDICTIONS_DIR + predictions_filename, header=0)
            apply.process(training_results, stat, predictions_filename, predictions_df)
            progress.increment_done()
    apply.process_end()
