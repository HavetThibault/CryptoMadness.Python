from ml_sdk.model.layers.output.layer_output import LayerOutput
from ml_sdk.model.layers.output.output_layer_builder import OutputLayerBuilder


class MultipleOutputLayerBuilder(OutputLayerBuilder):
    def __init__(self, output_layer_builders: list[OutputLayerBuilder]):
        self._output_layer_builders = output_layer_builders

    def create_output_layer(self, input) -> list[LayerOutput]:
        output_layers = []
        for layer_builder in self._output_layer_builders:
            output_layers += layer_builder.create_output_layer(input)
        return output_layers