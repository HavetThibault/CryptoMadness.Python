import os.path

from skopt import gp_minimize

from crypto.mlp_experiences.mlp_process import get_mlp_regression_model_creator
from crypto.mlp_experiences.regression1_params import Regression1Params
from ml_sdk.analyze.models.val_loss_models_analyzer import ValLossModelsAnalyzer
from ml_sdk.training.cache_training_runner import CacheTrainingRunner
from ml_sdk.training.callbacks import get_plateau_sheduler, get_early_stopping
from ml_sdk.training.models_results_infos import ModelsResultsInfos
from ml_sdk.training.val_loss_error_calculator import ValLossErrorCalculator


def optimize_mlp_h1():
    model_creator = get_mlp_regression_model_creator(
        Regression1Params.TRAIN_PATH,
        Regression1Params.VAL_PATH,
        Regression1Params.TOTAL_INPUTS,
        Regression1Params.MODEL_NAME,
        Regression1Params.BATCH_SIZE,
        Regression1Params.OUTPUTS,
        Regression1Params.H1_INIT_LR
    )
    models_results = ModelsResultsInfos.get_instance(
        Regression1Params.H1_DEST_DIR + model_creator.get_model_name() + ModelsResultsInfos.FILE_EXT,
        Regression1Params.H1_DEST_DIR)
    models_analyzer = ValLossModelsAnalyzer()
    callbacks = [
        get_plateau_sheduler(Regression1Params.MIN_DELTA, Regression1Params.LR_PATIENCE, Regression1Params.LR_FACTOR),
        get_early_stopping(Regression1Params.MIN_DELTA, Regression1Params.STOP_PATIENCE)
    ]
    training_runner = CacheTrainingRunner(
        model_creator,
        models_results,
        models_analyzer,
        Regression1Params.REPEAT,
        Regression1Params.ITERATIONS,
        callbacks,
        'mean_squared_error',
        None,
        ValLossErrorCalculator()
    )
    gp_minimize(
        training_runner.get_training_metric,
        [(1, 130)],
        n_calls=100,
        random_state=10,
        verbose=False)


def optimize_mlp_h2():
    model_creator = get_mlp_regression_model_creator(
        Regression1Params.TRAIN_PATH,
        Regression1Params.VAL_PATH,
        Regression1Params.TOTAL_INPUTS,
        Regression1Params.MODEL_NAME,
        Regression1Params.BATCH_SIZE,
        Regression1Params.OUTPUTS,
        Regression1Params.H2_INIT_LR
    )
    models_results = ModelsResultsInfos.get_instance(
        Regression1Params.H2_DEST_DIR + model_creator.get_model_name() + ModelsResultsInfos.FILE_EXT,
        Regression1Params.H2_DEST_DIR)
    models_analyzer = ValLossModelsAnalyzer()
    callbacks = [
        get_plateau_sheduler(Regression1Params.MIN_DELTA, Regression1Params.LR_PATIENCE, Regression1Params.LR_FACTOR),
        get_early_stopping(Regression1Params.MIN_DELTA, Regression1Params.STOP_PATIENCE)
    ]
    training_runner = CacheTrainingRunner(
        model_creator,
        models_results,
        models_analyzer,
        Regression1Params.REPEAT,
        Regression1Params.ITERATIONS,
        callbacks,
        'mean_squared_error',
        None,
        ValLossErrorCalculator()
    )
    gp_minimize(
        training_runner.get_training_metric,
        [(1, 130), (1, 130)],
        n_calls=100,
        random_state=10,
        verbose=False
        )
