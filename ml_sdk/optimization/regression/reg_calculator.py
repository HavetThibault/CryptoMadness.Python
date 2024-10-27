from typing import Any, Optional

from statsmodels.regression.linear_model import RegressionResults, RegressionModel


class RegCalculator:
    def __init__(self):
        self._model: Optional[RegressionModel] = None
        self._reg_results: Optional[RegressionResults] = None

    def get_model(self):
        return self._model

    def get_reg_results(self):
        return self._reg_results

    def predict(self, x) -> Any:
        if self._model is None or self._reg_results is None:
            raise Exception('Call "init_model_and_result" method before calling "predict" method.')
        return self._model.predict(params=self._reg_results.params, exog=x)

    def init_get_result(self, x, y) -> RegressionResults:
        self.init_model_and_result(x, y)
        return self._reg_results

    def init_get_model(self, x, y) -> RegressionModel:
        self.init_model_and_result(x, y)
        return self._model

    def init_model_and_result(self, x, y):
        raise Exception('_create_model should be implemented')
