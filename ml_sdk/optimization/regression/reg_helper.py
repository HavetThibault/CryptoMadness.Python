from typing import Optional

import pandas as pd

from ml_sdk.optimization.regression.ds_transformer import DSTransformer
from ml_sdk.optimization.regression.reg_calculator import RegCalculator
from ml_sdk.optimization.regression.reg_distinct_cols_frame_finder import RegDistinctColsFrameFinder
from ml_sdk.optimization.regression.reg_frame_finder import CONST_COL
from ml_sdk.optimization.regression.reg_results_interpret import RegResultsInterpret


# First significant model with the biggest number of distinct col
def first_significant_best_frame(x: pd.DataFrame, dependent_var: pd.DataFrame, ds_transformer: DSTransformer,
                                 reg_calc: RegCalculator, results_interpret: RegResultsInterpret,
                                 verbose: bool) -> dict[int, Optional[tuple[float, list[str]]]]:
    transformed_ds = ds_transformer.transform(x)
    reg_frame_finder = RegDistinctColsFrameFinder(min_distinct_cols=1)
    return reg_frame_finder.get_best_frame(transformed_ds.copy(), dependent_var, reg_calc, results_interpret,
                                           verbose)


def step_best_regression_model(x: pd.DataFrame, y, reg_calculator: RegCalculator, results_interpret: RegResultsInterpret):
    df = pd.DataFrame(x)
    best_frames = None
    best_value = None
    selected_cols = df.columns.to_list()
    while len(df.columns) >= 2:
        results = reg_calculator.init_get_result(df, y)
        value = results_interpret.interpret(results, best_value)
        if value is not None and (best_value is None or value >= best_value):
            best_frames = selected_cols
            best_value = value
        max_pvalue = None
        max_pvalue_index = -1
        for i, pvalue in enumerate(results.pvalues):
            if df.columns[i] != CONST_COL:
                if max_pvalue_index == -1:
                    max_pvalue_index = i
                    max_pvalue = pvalue
                else:
                    if pvalue > max_pvalue:
                        max_pvalue = pvalue
                        max_pvalue_index = i
        selected_cols = df.columns.delete(max_pvalue_index).to_list()
        df = df[selected_cols]
    return best_frames

