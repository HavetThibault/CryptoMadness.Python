from typing import Optional, Any

from statsmodels.regression.linear_model import RegressionResults

from ml_sdk.optimization.regression.reg_results_interpret import RegResultsInterpret


class RSquaredResultsInterpret(RegResultsInterpret):
    def __init__(self, max_r, coef_pvalue=0.05):
        super(RSquaredResultsInterpret, self).__init__()
        self._coef_pvalue = coef_pvalue
        self._max_r = max_r

    def interpret(self, results: RegressionResults, best_value) -> Optional[Any]:
        if best_value is None or self._max_r > results.rsquared_adj > best_value:
            for pvalue in results.pvalues:
                if pvalue > self._coef_pvalue:
                    return None
            return results.rsquared_adj
        return None