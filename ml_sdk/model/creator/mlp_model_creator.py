from keras import Model

from ml_sdk.model.creator.model_creator import ModelCreator


class MLPModelCreator(ModelCreator):
    def __init__(self, model_name, batch_size, output_builder, ds_creator_builder, layer_name_giver):
        super(MLPModelCreator, self).__init__(
            model_name, batch_size, output_builder, ds_creator_builder, layer_name_giver)

    def create_untrained_model(self, params_set: list, optimizer=None, loss=None) -> Model:
        pass