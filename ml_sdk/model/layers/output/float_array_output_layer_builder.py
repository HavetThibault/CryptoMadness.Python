from keras.layers import Dense

from layer_output import LayerOutput
from output_layer_builder import OutputLayerBuilder


class FloatArrayOutputLayerBuilder(OutputLayerBuilder):
    def __init__(self, output_name, layer_name, array_len, activation=None):
        self._array_len = array_len
        if activation is None:
            self._activation = 'linear'
        else:
            self._activation = activation
        self._output_name: str = output_name
        self._layer_name: str = layer_name

    def create_output_layer(self, input) -> list[LayerOutput]:
        return [
            LayerOutput(
                self._output_name,
                Dense(
                    units=self._array_len,
                    activation=self._activation,
                    name='dense_' + self._output_name)(input),
                self._layer_name)
        ]
