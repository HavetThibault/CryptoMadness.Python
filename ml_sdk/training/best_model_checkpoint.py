import math
import os

import tensorflow as tf

from ml_sdk.training.error_calculator import ErrorCalculator
from ml_sdk.training.model_filename_mgmt import get_model_filename
from ml_sdk.training.training_epoch_stats import TrainingEpochStats


class BestModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, serialization_dir, model_name: str, error_calc: ErrorCalculator, verbose=0, weights_only=True):
        super(BestModelCheckpoint, self).__init__()
        self._save_weights_only = weights_only
        self._serialization_dir = serialization_dir
        self._verbose = verbose
        self._model_name = model_name
        self._best_value = None
        self._best_stats = None
        self._best_model_file_path = None
        self._error_calc = error_calc
        self._last_learning_rate = None

    def get_best_model_file_path(self):
        return self._best_model_file_path

    def on_train_begin(self, logs=None):
        self._best_model_file_path = None
        self._best_value = None
        self._best_stats = None
        self._last_learning_rate = None

    def on_epoch_begin(self, epoch, logs=None):
        optimizer = self.model.optimizer
        if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            learning_rate = float(optimizer.learning_rate(optimizer.iterations))
        else:
            learning_rate = float(optimizer.learning_rate)
        if self._last_learning_rate is not None and not math.isclose(learning_rate, self._last_learning_rate):
            if self._verbose:
                print('Continuing from best weights...')
            self.model.load_weights(self._serialization_dir + self._best_model_file_path)
        self._last_learning_rate = learning_rate

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if loss is None or val_loss is None:
            return

        current_value = self._error_calc.get_error(loss, val_loss)
        if current_value is None:
            if self._verbose > 0:
                print(f"\nEpoch {epoch + 1}: Custom metric did not improve from {self._best_value}")
            return

        if self._best_value is None or current_value < self._best_value:
            if self._verbose > 0:
                print(f"\nEpoch {epoch + 1}: Custom metric improved from {self._best_value} to "
                      f"{current_value}, saving model...")
            self._best_value = current_value
            self._best_stats = TrainingEpochStats(epoch, loss, val_loss)

            model_file_path = self._serialization_dir + get_model_filename(self._model_name, self._best_stats,
                                                                          self._save_weights_only)
            self.model.save(model_file_path, overwrite=True)
            if self._best_model_file_path is not None:
                os.remove(self._serialization_dir + self._best_model_file_path)
            self._best_model_file_path = get_model_filename(self._model_name, self._best_stats, self._save_weights_only)
        else:
            if self._verbose > 0:
                print(f"\nEpoch {epoch + 1}: Custom metric did not improve from {self._best_value}")

    def on_train_end(self, logs=None):
        if self._verbose > 0:
            print(f"\nTraining completed. Best custom metric value: {self._best_stats}")
