from typing import Optional, Any

from model.optimization.regression.reg_results_interpret import RegResultsInterpret
from statsmodels.regression.linear_model import RegressionResults


class AICRSquaredResultInterpret(RegResultsInterpret):
    def __init__(self, coef_pvalue=0.05):
        super(AICRSquaredResultInterpret, self).__init__()
        self._coef_pvalue = coef_pvalue

    def interpret(self, results: RegressionResults, best_value) -> Optional[Any]:
        if best_value is None or results.aic * results.rsquared_adj > best_value:
            # for pvalue in results.pvalues:
            #     if pvalue > self._coef_pvalue:
            #         return None
            return results.aic * results.rsquared_adj
        return None
