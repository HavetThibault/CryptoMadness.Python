import os

from crypto.analyse.labels_and_preds_crypto_accuracy import LabelsAndPredsCryptoAccuracy
from crypto.mlp_experiences.mlp_process import regression_mlp_process, get_mlp_regression_model_creator
from crypto.mlp_experiences.regression1_params import Regression1Params
from helper_sdk.pandas_helper import df_min_max
from helper_sdk.work_progress_state import WorkProgressState, default_progress_display
from ml_sdk.analyze.analyze import apply_labels_and_preds
from ml_sdk.dataset.cook.ds_source_file import read_dataset
from ml_sdk.training.models_results_infos import ModelsResultsInfos
from ml_sdk.training.parameter import Parameter


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


def regression_metrics(dest_dir: str, init_lr):
    ds = read_dataset(Regression1Params.SRC_DS_PATH, header=0)
    close_cols = []
    for col in list(ds.columns):
        if col.startswith('Close') and not col.endswith('prediction'):
            close_cols.append(col)
    close_min, close_max = df_min_max(ds[close_cols])
    print(f'Close range: [{close_min}, {close_max}]')

    model_creator = get_mlp_regression_model_creator(
        Regression1Params.TRAIN_PATH,
        Regression1Params.VAL_PATH,
        Regression1Params.TOTAL_INPUTS,
        Regression1Params.MODEL_NAME,
        Regression1Params.BATCH_SIZE,
        Regression1Params.OUTPUTS,
        init_lr
    )

    progress = WorkProgressState()
    progress.add_listener(default_progress_display)
    metrics_path = ModelsResultsInfos.static_get_metrics_path(dest_dir, model_creator.get_model_name())
    if not os.path.exists(metrics_path):
        apply_labels_and_preds(
            model_creator,
            dest_dir,
            progress,
            LabelsAndPredsCryptoAccuracy(Regression1Params.VAL_PATH, metrics_path, close_min, close_max)
        )
    else:
        print(f'Already found a folder with the metrics : {metrics_path}')


def regression1_mlp():
    params_sets = [[20 + i * 15] for i in range(3)]
    regression_mlp(Regression1Params.H1_DEST_DIR, Regression1Params.H1_INIT_LR, params_sets)


def regression1_mlp_ext():
    params_sets = [[Parameter('hidden_layer1', 65 + i * 15)] for i in range(3)]
    regression_mlp(Regression1Params.h1_ext_dest_dir, Regression1Params.H1_INIT_LR, params_sets)


def regression1_metrics():
    regression_metrics(Regression1Params.H1_DEST_DIR, Regression1Params.H1_INIT_LR)


def regression2_mlp():
    params_sets = [[20 + i * 15, 10 + i * 15] for i in range(3)]
    regression_mlp(Regression1Params.H2_DEST_DIR, Regression1Params.H2_INIT_LR, params_sets)


def regression2_mlp_ext():
    params_sets = [[65 + i * 15, 50 + i * 15] for i in range(3)]
    regression_mlp(Regression1Params.h2_ext_dest_dir, Regression1Params.H2_INIT_LR, params_sets)


def regression2_mlp_metrics():
    regression_metrics(Regression1Params.H2_DEST_DIR, Regression1Params.H2_INIT_LR)
