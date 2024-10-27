
from ml_sdk.model.layers.output.layer_output import LayerOutput


class OutputLayerBuilder:
    def create_output_layer(self, input) -> list[LayerOutput]:
        raise NotImplementedError()
