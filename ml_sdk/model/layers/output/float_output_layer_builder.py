from keras.layers import Dense

from ml_sdk.model.layers.output.layer_output import LayerOutput
from ml_sdk.model.layers.output.output_layer_builder import OutputLayerBuilder


class FloatOutputLayerBuilder(OutputLayerBuilder):
    def __init__(self, output_name, layer_name, activation=None):
        if activation is None:
            self._activation = 'linear'
        else:
            self._activation = activation
        self._output_name = output_name
        self._layer_name: str = layer_name

    def create_output_layer(self, input) -> list[LayerOutput]:
        return [
            LayerOutput(
                self._output_name,
                Dense(units=1, activation=self._activation, name=self._layer_name)(input),
                self._layer_name)
        ]