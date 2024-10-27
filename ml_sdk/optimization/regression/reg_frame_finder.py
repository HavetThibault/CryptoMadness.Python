from typing import Optional

import pandas as pd
from model.optimization.regression.reg_calculator import RegCalculator
from model.optimization.regression.reg_results_interpret import RegResultsInterpret

CONST_COL = 'const'


class RegFrameFinder:
    def get_best_frame(self, x: pd.DataFrame, y, reg_calculator: RegCalculator, interpret: RegResultsInterpret) \
            -> dict[int, Optional[tuple[float, list[str]]]]:
        pass
