from statsmodels.regression.linear_model import GLS

from ml_sdk.optimization.regression.reg_calculator import RegCalculator


class GLSRegCalculator(RegCalculator):
    def init_model_and_result(self, x, y):
        self._model = GLS(y, x)
        self._reg_results = self._model.fit()
