from model.optimization.regression.reg_calculator import RegCalculator
from statsmodels.regression.linear_model import GLS


class GLSRegCalculator(RegCalculator):
    def init_model_and_result(self, x, y):
        self._model = GLS(y, x)
        self.reg_results = self._model.fit()
