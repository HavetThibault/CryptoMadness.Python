class LayerNameGiver:
    NAME_REGEX = '^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$'

    def __init__(self, input_prefix, output_prefix):
        self._input_prefix = input_prefix
        self._output_prefix = output_prefix

    def give_output_names(self, names: list[str]) -> list[tuple[str, str]]:
        return [(name, self.matching_output_name(name)) for name in names]

    def matching_output_name(self, name: str) -> str:
        return self._output_prefix + self.matching_name(name)

    def give_input_names(self, names: list[str]) -> list[tuple[str, str]]:
        return [(name, self.matching_input_name(name)) for name in names]

    def matching_input_name(self, name: str) -> str:
        return self._input_prefix + self.matching_name(name)

    def matching_name(self, name: str) -> str:
        raise NotImplementedError()