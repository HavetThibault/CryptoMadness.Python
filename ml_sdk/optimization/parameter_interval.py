

class ParameterInterval:
    def __init__(self, min, step_nbr, step):
        self._min = min
        self._step_nbr = step_nbr
        self._step = step

    def generate_interval(self) -> list[float]:
        values = []
        for i in range(self._step_nbr):
            values.append(i * self._step + self._min)
        return values
