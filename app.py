import pandas as pd

from ml_sdk.dataset.file.file_record_field_type import RecordFieldType
from ml_sdk.dataset.file.file_record_struct_builder import FileRecordStructBuilder
from ml_sdk.dataset.model_feeder.mlp_ds_creator_builder import MLPDsCreatorBuilder
from ml_sdk.model.creator.mlp_model_creator import MLPModelCreator
from ml_sdk.model.layers.end_rm_name_giver import EndRmNameGiver
from ml_sdk.model.layers.output.category_output_layer_builder import CategoryOutputLayerBuilder
from ml_sdk.training.best_model_checkpoint import BestModelCheckpoint
from ml_sdk.training.callbacks import get_plateau_sheduler, get_early_stopping
from ml_sdk.training.training_memory import TrainingMemory
from ml_sdk.training.trainings_runner import run_trainings
from ml_sdk.training.val_loss_error_calculator import ValLossErrorCalculator

if __name__ == '__main__':
    train_path = 'C:/Users/Thibault/Downloads/BTC/data/btc_train.csv'
    val_path = 'C:/Users/Thibault/Downloads/BTC/data/btc_val.csv'
    dest_dir = 'C:/Users/Thibault/Downloads/BTC/Models/'
    weights_dir = 'C:/Users/Thibault/Downloads/BTC/Models/Temp/'
    archived_dir = 'C:/Users/Thibault/Downloads/BTC/Models/Archived/'
    model_name = 'mlp_v1'
    output_name = 'outputs'

    inputs = 60 * 2
    outputs = 2
    iterations = 500
    batch_size = 512
    min_delta = 0.0005
    lr_patience = 5
    lr_factor = 0.1
    stop_patience = 15
    init_lr = 0.1
    repeat = 3

    train_file = pd.read_csv(train_path)
    record_struct = []
    for i in range(len(train_file.columns)):
        record_struct.append(RecordFieldType(RecordFieldType.FLOAT))

    memory_filepath = dest_dir + model_name + TrainingMemory.FILE_EXT
    error_calculator = ValLossErrorCalculator()
    ds_creator_builder = MLPDsCreatorBuilder(
        train_path,
        val_path,
        FileRecordStructBuilder(record_struct),
        batch_size,
        outputs)

    layer_name_giver = EndRmNameGiver('', '')
    layer_output_name = layer_name_giver.matching_output_name(output_name)

    output_layers_builder = CategoryOutputLayerBuilder(output_name, layer_output_name, outputs)

    params_sets = [[50]]
    model_creator = MLPModelCreator(
        model_name,
        batch_size,
        output_layers_builder,
        ds_creator_builder,
        layer_name_giver,
        inputs,
        init_lr)

    callbacks = [
        get_plateau_sheduler(min_delta, lr_patience, lr_factor),
        get_early_stopping(min_delta, stop_patience, verbose=1)
    ]

    run_trainings(
        params_sets,
        model_creator,
        memory_filepath,
        weights_dir,
        iterations,
        archived_dir,
        error_calculator,
        repeat,
        additional_callbacks=callbacks,
        verbose=1,
        on_cpu=True
    )
