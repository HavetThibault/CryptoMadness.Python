import os

from mlp_process import classification_mlp_process, regression_mlp_process

if __name__ == '__main__':
    inputs = 240
    total_inputs = inputs * 2
    after = 2
    outputs = 1
    iterations = 500
    batch_size = 64
    min_delta = 0.0005
    lr_patience = 0
    lr_factor = 0.1
    stop_patience = 3
    init_lr = 0.1
    repeat = 1
    version = 1
    intervals = 0
    normalized = 1

    root = 'E:/BTC/'
    # root = 'C:/Users/Maison/Documents/Thibault/BTC/'
    str_id = f'b{inputs}_a{after}_i{intervals}_v{version}_n{normalized}'
    train_path = root + f'TrainVal/btc_{str_id}_train.csv'
    val_path = root + f'TrainVal/btc_{str_id}_val.csv'
    dest_dir = root + f'Models_{str_id}_1/'
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    print('Created folder ' + dest_dir)
    model_name = f'mlp_v{version}'
    output_name = 'outputs'

    params_sets = [[20 + i * 15] for i in range(3)]

    regression_mlp_process(
        train_path,
        val_path,
        dest_dir,
        model_name,
        batch_size,
        outputs,
        output_name,
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

    dest_dir = root + f'Models_{str_id}_2/'
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    print('Created folder ' + dest_dir)
    model_name = f'mlp_v{version}'
    params_sets = [[20 + i * 15, 10 + i * 15] for i in range(3)]

    classification_mlp_process(
        train_path,
        val_path,
        dest_dir,
        model_name,
        batch_size,
        outputs,
        output_name,
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
