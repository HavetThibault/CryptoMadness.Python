import os

from crypto.analyse.labels_and_preds_crypto_accuracy import LabelsAndPredsCryptoAccuracy
from crypto.mlp_experiences.mlp_process import regression_mlp_process, get_mlp_regression_model_creator
from crypto.mlp_experiences.regression1_params import Regression1Params
from helper_sdk.pandas_helper import df_min_max
from helper_sdk.work_progress_state import WorkProgressState, default_progress_display
from ml_sdk.analyze.analyze import apply_labels_and_preds
from ml_sdk.dataset.cook.ds_source_file import read_dataset
from ml_sdk.training.models_results_infos import ModelsResultsInfos


def regression_mlp(dest_dir: str, init_lr, params_sets):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        print('Created folder ' + dest_dir)

    regression_mlp_process(
        Regression1Params.TRAIN_PATH,
        Regression1Params.VAL_PATH,
        dest_dir,
        Regression1Params.MODEL_NAME,
        Regression1Params.BATCH_SIZE,
        Regression1Params.OUTPUTS,
        params_sets,
        Regression1Params.REPEAT,
        Regression1Params.TOTAL_INPUTS,
        init_lr,
        Regression1Params.MIN_DELTA,
        Regression1Params.LR_PATIENCE,
        Regression1Params.LR_FACTOR,
        Regression1Params.STOP_PATIENCE,
        Regression1Params.ITERATIONS
    )

