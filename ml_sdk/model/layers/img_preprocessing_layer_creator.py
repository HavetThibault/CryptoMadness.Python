from keras.layers import RandomFlip, RandomRotation, Rescaling


class ImgPreProcessingLayerCreator:
    HORIZONTAL_VERTICAL_FLIP = 'horizontal_and_vertical'

    def __init__(self, rescale_factor: float = None, flip: str = None, rotation: float = None):
        self._flip = flip
        self._rescale = rescale_factor
        self._rotation = rotation

    def create_connected_layers(self, inputs):
        x = inputs
        if self._rescale is not None:
            x = Rescaling(self._rescale)(x)

        if self._flip is not None:
            x = RandomFlip(self._flip)(x)

        if self._rotation is not None:
            x = RandomRotation(self._rotation)(x)

        return x
