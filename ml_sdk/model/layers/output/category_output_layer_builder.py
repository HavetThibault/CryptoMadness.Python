from keras.layers import Dense

from ml_sdk.model.layers.output.layer_output import LayerOutput
from ml_sdk.model.layers.output.output_layer_builder import OutputLayerBuilder


class CategoryOutputLayerBuilder(OutputLayerBuilder):
    def __init__(self, output_name, layer_name, categories_nbr):
        self._categories_nbr = categories_nbr
        self._output_name: str = output_name
        self._layer_name: str = layer_name

    def create_output_layer(self, input) -> list[LayerOutput]:
        return [
            LayerOutput(
                self._output_name,
                Dense(units=self._categories_nbr, activation='softmax')(input),
                self._layer_name)
        ]
