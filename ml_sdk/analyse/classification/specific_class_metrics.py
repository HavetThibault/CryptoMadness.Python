from ml_sdk.analyse.classification.success_rate import SuccessRate


class SpecificClassMetrics:
    def __init__(self, intervals: int):
        self._success_rate = SuccessRate()
        self._intervals_success = [SuccessRate() for _ in range(intervals)]

    def add_right(self, interval):
        self._success_rate.add_right()
        self._intervals_success[interval].add_right()

    def add_wrong(self, interval):
        self._success_rate.add_wrong()
        self._intervals_success[interval].add_wrong()

    def get_rates(self) -> list[float]:
        return [success_rate.get_rate() for success_rate in self._intervals_success]
