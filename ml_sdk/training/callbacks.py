import tensorflow as tf


def get_plateau_sheduler(min_delta, patience=5, factor=0.1):
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience,
        verbose=0,
        mode='min',
        min_delta=min_delta,
        min_lr=0.00001)


def get_early_stopping(min_delta, patience=5, verbose=0):
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        mode='min',
        baseline=None,
        restore_best_weights=False
    )
