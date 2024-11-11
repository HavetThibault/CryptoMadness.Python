from statsmodels.regression.linear_model import OLS

from ml_sdk.optimization.regression.reg_calculator import RegCalculator


class OLSRegCalculator(RegCalculator):
    def init_model_and_result(self, x, y):
        self._model = OLS(y, x)
        self._reg_results = self._model.fit()
