import re

from ml_sdk.model.layers.layer_name_giver import LayerNameGiver


class EndRmNameGiver(LayerNameGiver):
    def __init__(self, input_prefix, output_prefix):
        super(EndRmNameGiver, self).__init__(input_prefix, output_prefix)

    def matching_name(self, name: str) -> str:
        valid_name = name
        while not re.match(LayerNameGiver.NAME_REGEX, valid_name):
            valid_name = valid_name[:-1]
        return valid_name