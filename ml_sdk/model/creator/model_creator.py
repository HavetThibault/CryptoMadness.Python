from typing import Optional

import keras.backend
import tensorflow as tf
from keras import Model
from keras.callbacks import History
from keras.optimizers import Adam

from helper_sdk.exception_helper import format_exception_to_str
from ml_sdk.dataset.model_feeder.ds_creator_builder import DsCreatorBuilder
from ml_sdk.model.layers.layer_name_giver import LayerNameGiver
from ml_sdk.model.layers.output.output_layer_builder import OutputLayerBuilder
from ml_sdk.training.parameter import Parameter


class ModelCreator:
    def __init__(self, model_name, batch_size, output_builder, ds_creator_builder, layer_name_giver):
        self._model_name: str = model_name
        self._batch_size: int = batch_size
        self._output_builder: OutputLayerBuilder = output_builder
        self._ds_creator_builder: DsCreatorBuilder = ds_creator_builder
        self._layer_name_giver: LayerNameGiver = layer_name_giver

    def get_model_name(self) -> str:
        return self._model_name

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_ds_creator_builder(self) -> DsCreatorBuilder:
        return self._ds_creator_builder

    def create_untrained_model(self, params_set: list[Parameter], optimizer=None, loss=None) -> Model:
        raise NotImplementedError('Method "_create_untrained_model" not implemented')

    def create_and_train_model(self, params_set: list, iterations_nbr, callbacks, give_up_cpu, on_cpu, optimizer=None,
                               starts_weights_file=None, start_model=None, loss=None, verbose=0) \
            -> Optional[History]:
        if starts_weights_file is not None:
            model = self.load_model_from_weights(params_set, starts_weights_file, optimizer, loss)
        elif start_model is not None:
            model = start_model
        else:
            model = self.create_untrained_model(params_set, optimizer, loss)
        (train_ds, train_len), (val_ds, val_len) = self._ds_creator_builder.get_train_val_ds(iterations_nbr)

        batch_per_epoch = int(train_len / self._batch_size)
        val_batch_per_epoch = int(val_len / self._batch_size)

        try:
            if not on_cpu:
                try:
                    return model.fit(
                        x=train_ds,
                        validation_data=val_ds,
                        callbacks=callbacks,
                        epochs=iterations_nbr,
                        steps_per_epoch=batch_per_epoch,
                        validation_steps=val_batch_per_epoch,
                        verbose=verbose)
                except tf.errors.ResourceExhaustedError:
                    print('Got tf.errors.ResourceExhaustedError.')
                    if give_up_cpu:
                        return None
                    print('Running on CPU...')
                    # Pay attention, tensorflow does not say anything if the device is not found, and will
                    # run on default mode
                    with tf.device('/CPU:0'):
                        return model.fit(
                            x=train_ds,
                            validation_data=val_ds,
                            callbacks=callbacks,
                            epochs=iterations_nbr,
                            steps_per_epoch=batch_per_epoch,
                            validation_steps=val_batch_per_epoch,
                            verbose=verbose)
            else:
                with tf.device('/CPU:0'):
                    return model.fit(
                        x=train_ds,
                        validation_data=val_ds,
                        callbacks=callbacks,
                        epochs=iterations_nbr,
                        steps_per_epoch=batch_per_epoch,
                        validation_steps=val_batch_per_epoch,
                        verbose=verbose)
        except Exception as e:
            print(format_exception_to_str(e))
        finally:
            keras.backend.clear_session()

    def load_model_from_weights(self, params_set: list, weights_file: str, optimizer=None, loss=None) -> Model:
        try:
            model = self.create_untrained_model(params_set, optimizer, loss)
            model.load_weights(weights_file)
        except tf.errors.ResourceExhaustedError:
            print('No enough space on GPU, running on CPU...')
            with tf.device('/CPU:0'):
                model = self.create_untrained_model(params_set, optimizer, loss)
                model.load_weights(weights_file)
        return model

    @staticmethod
    def load_model(model_file) -> Model:
        return tf.keras.models.load_model(model_file)

    @staticmethod
    def _compile_model(model, optimizer, loss):
        if optimizer is None:
            optimizer = Adam()
        if loss is None:
            loss = 'mean_squared_error'
        model.compile(optimizer=optimizer, loss=loss)
        return model
