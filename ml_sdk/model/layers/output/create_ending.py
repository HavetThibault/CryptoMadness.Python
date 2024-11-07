from typing import Any, Optional

from keras.layers import Dense

from ml_sdk.model.layers.output.layer_output import LayerOutput
from ml_sdk.model.layers.output.output_layer_builder import OutputLayerBuilder


def create_relu_layers(input, output_builder: OutputLayerBuilder, neurons: list[int]) -> dict[str, Any]:
    neurons_activation = [(neuron, 'relu') for neuron in neurons]
    return create_layers(input, output_builder, neurons_activation)


def create_sigmoid_layers(input, output_builder: OutputLayerBuilder, neurons: list[int]) -> dict[str, Any]:
    neurons_activation = [(neuron, 'sigmoid') for neuron in neurons]
    return create_layers(input, output_builder, neurons_activation)


def create_layers(input, output_builder: OutputLayerBuilder,
                  neurons_activation: list[tuple[int, Any]]):
    dense_layers_output = input
    for neurons, activation in neurons_activation:
        dense_layers_output = Dense(units=neurons, activation=activation)(dense_layers_output)
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
