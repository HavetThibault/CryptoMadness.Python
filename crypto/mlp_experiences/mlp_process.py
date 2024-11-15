from helper_sdk.csv_helper import read_headers
from helper_sdk.work_progress_state import WorkProgressState, default_progress_display, get_default_progress_state
from ml_sdk.analyze.analyze import analyze
from ml_sdk.analyze.reg_model_labels_and_preds_instantiator import RegModelLabelsAndPredsInstantiator
from ml_sdk.dataset.file.file_record_field_type import RecordFieldType
from ml_sdk.dataset.file.file_record_struct_builder import FileRecordStructBuilder
from ml_sdk.dataset.model_feeder.mlp_ds_creator_builder import MLPDsCreatorBuilder
from ml_sdk.model.creator.mlp_model_creator import MLPModelCreator
from ml_sdk.model.layers.end_rm_name_giver import EndRmNameGiver
from ml_sdk.model.layers.output.category_output_layer_builder import CategoryOutputLayerBuilder
from ml_sdk.model.layers.output.float_output_layer_builder import FloatOutputLayerBuilder
from ml_sdk.training.callbacks import get_plateau_sheduler, get_early_stopping
from ml_sdk.training.models_results_infos import ModelsResultsInfos
from ml_sdk.training.predefined_training_runner import run_trainings
from ml_sdk.training.val_loss_error_calculator import ValLossErrorCalculator


def classification_mlp_process(
        train_path,
        val_path,
        dest_dir,
        model_name,
        batch_size,
        outputs,
        params_sets,
        repeat,
        total_inputs,
        init_lr,
        min_delta,
        lr_patience,
        lr_factor,
        stop_patience,
        iterations):
    columns = read_headers(train_path)
    record_struct = []
    for i in range(len(columns)):
        record_struct.append(RecordFieldType(RecordFieldType.FLOAT))

    memory_filepath = dest_dir + model_name + ModelsResultsInfos.FILE_EXT
    error_calculator = ValLossErrorCalculator()
    ds_creator_builder = MLPDsCreatorBuilder(
        train_path,
        val_path,
        FileRecordStructBuilder(record_struct),
        batch_size,
        outputs)

    layer_name_giver = EndRmNameGiver('', '')
    layer_output_name = layer_name_giver.matching_output_name(MLPModelCreator.OUTPUT_NAME)

    output_layers_builder = CategoryOutputLayerBuilder(MLPModelCreator.OUTPUT_NAME, layer_output_name, outputs)

    model_creator = MLPModelCreator(
        model_name,
        batch_size,
        output_layers_builder,
        ds_creator_builder,
        layer_name_giver,
        total_inputs,
        init_lr)

    callbacks = [
        get_plateau_sheduler(min_delta, lr_patience, lr_factor),
        get_early_stopping(min_delta, stop_patience)
    ]

    progress = get_default_progress_state(len(params_sets) * repeat)
    run_trainings(
        params_sets,
        model_creator,
        memory_filepath,
        dest_dir,
        iterations,
        error_calculator,
        repeat,
        loss='categorical_crossentropy',
        additional_callbacks=callbacks,
        verbose=1
    )

    preds_instatiator = RegModelLabelsAndPredsInstantiator()
    analyze(model_creator, preds_instatiator, dest_dir, progress)


def get_mlp_regression_model_creator(
        train_path,
        val_path,
        total_inputs,
        model_name,
        batch_size,
        outputs,
        init_lr):
    columns = read_headers(train_path)
    record_struct = []
    for i in range(len(columns)):
        record_struct.append(RecordFieldType(RecordFieldType.FLOAT))

    layer_name_giver = EndRmNameGiver('', '')
    layer_output_name = layer_name_giver.matching_output_name(MLPModelCreator.OUTPUT_NAME)

    ds_creator_builder = MLPDsCreatorBuilder(
        train_path,
        val_path,
        FileRecordStructBuilder(record_struct),
        batch_size,
        outputs)

    output_layers_builder = FloatOutputLayerBuilder(MLPModelCreator.OUTPUT_NAME, layer_output_name)

    return MLPModelCreator(
        model_name,
        batch_size,
        output_layers_builder,
        ds_creator_builder,
        layer_name_giver,
        total_inputs,
        init_lr)


def regression_mlp_process(
        train_path,
        val_path,
        dest_dir,
        model_name,
        batch_size,
        outputs,
        params_sets,
        repeat,
        total_inputs,
        init_lr,
        min_delta,
        lr_patience,
        lr_factor,
        stop_patience,
        iterations):
    memory_filepath = dest_dir + model_name + ModelsResultsInfos.FILE_EXT
    error_calculator = ValLossErrorCalculator()
    model_creator = get_mlp_regression_model_creator(
        train_path,
        val_path,
        total_inputs,
        model_name,
        batch_size,
        outputs,
        init_lr
    )

    callbacks = [
        get_plateau_sheduler(min_delta, lr_patience, lr_factor),
        get_early_stopping(min_delta, stop_patience)
    ]

    progress = get_default_progress_state(len(params_sets) * repeat)
    run_trainings(
        params_sets,
        model_creator,
        memory_filepath,
        dest_dir,
        iterations,
        error_calculator,
        repeat,
        loss='mean_squared_error',
        additional_callbacks=callbacks,
        verbose=1
    )

    preds_instatiator = RegModelLabelsAndPredsInstantiator()
    analyze(model_creator, preds_instatiator, dest_dir, progress)