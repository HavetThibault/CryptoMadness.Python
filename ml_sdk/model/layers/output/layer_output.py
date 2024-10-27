class LayerOutput:
    def __init__(self, output_name, tf_output, layer_name):
        self._output_name: str = output_name
        self._tf_output = tf_output
        self._layer_name = layer_name

    def get_output_name(self):
        return self._output_name

    def get_tf_output(self):
        return self._tf_output

    def get_layer_name(self):
        return self._layer_name
