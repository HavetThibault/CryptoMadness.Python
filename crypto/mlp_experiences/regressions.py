import os

from crypto.analyse.labels_and_preds_crypto_accuracy import LabelsAndPredsCryptoAccuracy
from crypto.mlp_experiences.mlp_process import regression_mlp_process, get_mlp_regression_model_creator
from helper_sdk.pandas_helper import df_min_max
from helper_sdk.work_progress_state import WorkProgressState, default_progress_display
from ml_sdk.analyse.analyze import apply_labels_and_preds, METRICS_DIR
from ml_sdk.dataset.cook.ds_source_file import read_dataset

inputs = 60
total_inputs = inputs * 2
after = 2
outputs = 1
iterations = 2
batch_size = 64
min_delta = 15
lr_patience = 3
lr_factor = 0.1
stop_patience = 6
h1_init_lr = 0.01
h2_init_lr = 0.04
repeat = 3
version = 1
intervals = 0
normalized = 1


root = 'C:/MyProgs/Python/CryptoMadness.Python/data/'
str_id = f'b{inputs}_a{after}_i{intervals}_v{version}'
src_ds_path = root + f'CV_BTC_Data_{str_id}.csv'
str_id += f'_n{normalized}'
train_path = root + f'train_val/btc_{str_id}_train.csv'
val_path = root + f'train_val/btc_{str_id}_val.csv'
h1_dest_dir = root + f'Models_{str_id}_1/'
h2_dest_dir = root + f'Models_{str_id}_2/'
h1_ext_dest_dir = root + f'Models_{str_id}_1_ext/'
h2_ext_dest_dir = root + f'Models_{str_id}_2_ext/'
h1_metrics_dir = h1_dest_dir + METRICS_DIR
h2_metrics_dir = h2_dest_dir + METRICS_DIR
model_name = f'mlp_v{version}'


def regression_mlp(dest_dir: str, init_lr, params_sets):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        print('Created folder ' + dest_dir)

    regression_mlp_process(
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
        iterations
    )


def regression_metrics(dest_dir: str, init_lr, metrics_dir: str):
    ds = read_dataset(src_ds_path, header=0)
    close_cols = []
    for col in list(ds.columns):
        if col.startswith('Close') and not col.endswith('prediction'):
            close_cols.append(col)
    close_min, close_max = df_min_max(ds[close_cols])
    print(f'Close range: [{close_min}, {close_max}]')

    model_creator = get_mlp_regression_model_creator(
        train_path,
        val_path,
        total_inputs,
        model_name,
        batch_size,
        outputs,
        init_lr
    )

    progress = WorkProgressState()
    progress.add_listener(default_progress_display)
    if not os.path.exists(metrics_dir):
        os.mkdir(metrics_dir)
        apply_labels_and_preds(
            model_creator,
            dest_dir,
            progress,
            LabelsAndPredsCryptoAccuracy(val_path, metrics_dir, close_min, close_max)
        )
    else:
        print(f'Already found a folder with the metrics : {metrics_dir}')


def regression1_mlp():
    params_sets = [[20 + i * 15] for i in range(3)]
    regression_mlp(h1_dest_dir, h1_init_lr, params_sets)


def regression1_mlp_ext():
    params_sets = [[65 + i * 15] for i in range(3)]
    regression_mlp(h1_ext_dest_dir, h1_init_lr, params_sets)


def regression1_metrics():
    regression_metrics(h1_dest_dir, h1_init_lr, h1_metrics_dir)


def regression2_mlp():
    params_sets = [[20 + i * 15, 10 + i * 15] for i in range(3)]
    regression_mlp(h2_dest_dir, h2_init_lr, params_sets)


def regression2_mlp_ext():
    params_sets = [[65 + i * 15, 50 + i * 15] for i in range(3)]
    regression_mlp(h2_ext_dest_dir, h2_init_lr, params_sets)


def regression2_mlp_metrics():
    regression_metrics(h2_dest_dir, h2_init_lr, h2_metrics_dir)
