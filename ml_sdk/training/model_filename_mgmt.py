from ml_sdk.training.training_epoch_stats import TrainingEpochStats

WEIGHTS_FILE_EXTENSION = '.weights.h5'
EXPORT_MODEL_EXT = '.h5'
MODEL_FILE_EXTENSION = '.keras'


def _get_metrics_str(filename) -> str:
    model_name_len = len(filename.split('__')[0])
    if filename.endswith(WEIGHTS_FILE_EXTENSION):
        extension_len = len(WEIGHTS_FILE_EXTENSION)
    else:
        extension_len = len(MODEL_FILE_EXTENSION)
    return filename[model_name_len+2:-extension_len]


def get_training_epoch_stats(filename) -> TrainingEpochStats:
    stats = _get_metrics_str(filename).split('__')
    return TrainingEpochStats(int(stats[0]), float(stats[1]), float(stats[2]))


def get_model_filename(model_name: str, stats: TrainingEpochStats, weights_only) -> str:
    filename = f'{model_name}__{stats.get_epoch()}__{stats.get_loss():.5f}__{stats.get_val_loss():.5f}'
    if weights_only:
        return filename + WEIGHTS_FILE_EXTENSION
    return filename + MODEL_FILE_EXTENSION
