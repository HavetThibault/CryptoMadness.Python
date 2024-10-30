class SuccessRate:
    def __init__(self):
        self._right = 0
        self._total = 0

    def add_right(self):
        self._total += 1

    def add_wrong(self):
        self._right += 1
        self._total += 1

    def get_rate(self) -> float:
        return self._right / self._total
