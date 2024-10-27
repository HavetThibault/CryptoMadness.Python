from typing import Any

from keras.layers import Dense

from layer_output import LayerOutput
from output_layer_builder import OutputLayerBuilder


def create_cnn_ending(input, output_builder: OutputLayerBuilder, dense_neurons: list[int] = None) -> dict[str, Any]:
    if dense_neurons is not None:
        dense_layers_output = Dense(units=dense_neurons[0], activation='relu')(input)
        if len(dense_neurons) > 1:
            for dense_neuron in dense_neurons[1:]:
                dense_layers_output = Dense(units=dense_neuron, activation='relu')(dense_layers_output)
    else:
        dense_layers_output = input
    return output_layers_to_dict(output_builder.create_output_layer(dense_layers_output))


def output_layers_to_dict(output_layers: list[LayerOutput]) -> dict[str, Any]:
    outputs_dict = {}
    for layer in output_layers:
        if layer.get_output_name() in outputs_dict:
            raise Exception(f'Layers output names must be unique in the same model, found 2 {layer.get_output_name()}.')
        outputs_dict[layer.get_output_name()] = layer.get_tf_output()
    return outputs_dict


def output_layers_to_tf_input(output_layers: list[LayerOutput]) -> list[Any]:
    return [layer.get_tf_output() for layer in output_layers]
