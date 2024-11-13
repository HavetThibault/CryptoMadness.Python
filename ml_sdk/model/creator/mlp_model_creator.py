from keras import Model, Input
from keras.optimizers import Adam

from ml_sdk.model.creator.model_creator import ModelCreator
from ml_sdk.model.layers.output.create_ending import create_relu_layers
from ml_sdk.training.parameter import Parameter


class MLPModelCreator(ModelCreator):
    OUTPUT_NAME = 'outputs'
    INPUT_NAME = 'inputs'

    def __init__(self, model_name, batch_size, output_builder, ds_creator_builder, layer_name_giver, inputs, init_lr):
        super(MLPModelCreator, self).__init__(
            model_name, batch_size, output_builder, ds_creator_builder, layer_name_giver)
        self._inputs: int = inputs
        self._init_lr: float = init_lr

    def create_untrained_model(self, params_set: list[Parameter], optimizer=None, loss=None) -> Model:
        hidden_layers = [parameter.get_value() for parameter in params_set]

        input_layer = Input(shape=(self._inputs, ), name=self._layer_name_giver.matching_input_name(self.INPUT_NAME))
        end_layers_output = create_relu_layers(input_layer, self._output_builder, hidden_layers)

        input_layers = {self.INPUT_NAME: input_layer}
        model = Model(inputs=input_layers, outputs=end_layers_output)
        self._compile_model(model, Adam(learning_rate=self._init_lr), loss)
        return model

