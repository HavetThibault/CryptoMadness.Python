import os

from crypto.analyse.labels_and_preds_crypto_accuracy import LabelsAndPredsCryptoAccuracy
from crypto.mlp_experiences.mlp_process import get_mlp_regression_model_creator
from helper_sdk.pandas_helper import df_min_max
from helper_sdk.work_progress_state import WorkProgressState, default_progress_display, get_default_progress_state
from ml_sdk.analyze.analyze import apply_labels_and_preds
from ml_sdk.dataset.cook.ds_source_file import read_dataset
from ml_sdk.model.creator.model_creator import ModelCreator
from ml_sdk.training.models_results_infos import ModelsResultsInfos


def regression_metrics(normalized_src_ds_path: str, val_path: str, model_creator: ModelCreator, dest_dir: str):
    ds = read_dataset(normalized_src_ds_path, header=0)
    close_cols = []
    for col in list(ds.columns):
        if col.startswith('Close') and not col.endswith('prediction'):
            close_cols.append(col)
    close_min, close_max = df_min_max(ds[close_cols])
    print(f'Close range: [{close_min}, {close_max}]')

    progress = get_default_progress_state()
    metrics_path = ModelsResultsInfos.static_get_metrics_path(dest_dir, model_creator.get_model_name())
    if not os.path.exists(metrics_path):
        apply_labels_and_preds(
            model_creator,
            dest_dir,
            progress,
            LabelsAndPredsCryptoAccuracy(val_path, metrics_path, close_min, close_max)
        )
    else:
        print(f'Already found a folder with the metrics : {metrics_path}')