from ml_sdk.optimization.parameter_interval import ParameterInterval
from ml_sdk.training.parameter import Parameter


class ParametersMatrixGenerator:
    def __init__(self, parameter_intervals: list[ParameterInterval], levels=None, prefix_path=None):
        self._parameter_intervals = parameter_intervals
        self._levels = levels
        self._prefix_path = prefix_path

    def generate_all_matrix(self) -> list[list[float]]:
        params_values = []
        for param_interval in self._parameter_intervals:
            params_values.append(param_interval.generate_interval())

        params_nbr = len(params_values)
        comb_nbr = 1
        params_lens = []
        for i, params_value in enumerate(params_values):
            params_len = len(params_value)
            comb_nbr *= params_len
            if i > 0:
                for k in range(len(params_lens)):
                    params_lens[k] *= params_len
                params_lens.append(params_len)

        params_matrix = []
        print(f'Generating {comb_nbr} combination parameters matrix...')
        for i in range(comb_nbr):
            params_set = []
            for k in range(params_nbr - 1):
                param_index = (i // params_lens[k]) % len(params_values[k])
                params_set.append(params_values[k][param_index])
            param_index = i % len(params_values[params_nbr-1])
            params_set.append(params_values[params_nbr-1][param_index])
            params_matrix.append(params_set)
        return params_matrix


def params_set_to_str(params_set: list[Parameter]) -> str:
    params_str = ''
    for param in params_set:
        params_str += '{:0.1f}-'.format(param.get_value())
    return params_str[:-1]
