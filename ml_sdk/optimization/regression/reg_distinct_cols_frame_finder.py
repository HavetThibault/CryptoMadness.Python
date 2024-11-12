from typing import Optional

import pandas as pd

from helper_sdk.list_helper import contains
from ml_sdk.optimization.regression.reg_calculator import RegCalculator
from ml_sdk.optimization.regression.reg_frame_finder import RegFrameFinder, CONST_COL
from ml_sdk.optimization.regression.reg_results_interpret import RegResultsInterpret


class RegDistinctColsFrameFinder(RegFrameFinder):
    def __init__(self, original_columns=None, min_distinct_cols=None):
        super(RegDistinctColsFrameFinder, self).__init__()
        if min_distinct_cols is None:
            self._min_distinct_cols = len(self._find_distinct_columns(original_columns))
        else:
            self._min_distinct_cols = min_distinct_cols

    def get_best_frame(self, x: pd.DataFrame, y, reg_calculator: RegCalculator, interpret: RegResultsInterpret,
                       verbose=False) -> dict[int, Optional[tuple[float, list[str]]]]:
        best_value1 = None
        best_frames: dict[int, Optional[tuple[float, list[str]]]] = {}
        if self._min_distinct_cols <= 1:
            for i in range(len(x.columns)):
                if x.columns[i] != CONST_COL:
                    for k in range(2):
                        if k == 0:
                            sub_cols = [x.columns[i]] + [CONST_COL]
                        else:
                            sub_cols = x.columns[i]
                        results = reg_calculator.init_get_result(x[sub_cols], y)
                        interpret_result = interpret.interpret(results, best_value1)
                        if interpret_result is not None:
                            best_value1 = interpret_result
                            best_frames[1] = (best_value1, sub_cols)
        results = reg_calculator.init_get_result(x, y)
        interpret_result = interpret.interpret(results, None)
        if interpret_result is not None:
            best_frames[len(self._find_distinct_columns(x.columns))] = (interpret_result, x.columns.to_list())
        cols = x.columns.to_list()
        cols.remove(CONST_COL)
        print('--> Testing for these cols:', cols)
        print('--> Min distinct col:', self._min_distinct_cols)
        self._apply_distinct_sub_col_combinations(
            cols, x, y, reg_calculator, interpret, best_frames, verbose=verbose)
        return best_frames

    def _apply_distinct_sub_col_combinations(self, cols: list[str], x: pd.DataFrame, y, reg_calculator: RegCalculator,
                                             interpret: RegResultsInterpret, best_frames,
                                             step_ratio=None, progress=0, levels=0, verbose=False):
        if step_ratio is None:
            step_ratio = 1 / len(cols)
        else:
            step_ratio *= 1 / len(cols)
        for i in range(len(cols)):
            comb_copy = cols.copy()
            comb_copy.pop(i)

            distinct_cols = len(self._find_distinct_columns(comb_copy))
            if distinct_cols >= self._min_distinct_cols:
                if len(cols) > 1:
                    self._apply_distinct_sub_col_combinations(
                        comb_copy, x, y, reg_calculator, interpret,
                        best_frames, step_ratio, progress, levels + 1, verbose)

                for k in range(2):
                    if k == 0:
                        sub_cols = comb_copy + [CONST_COL]
                    else:
                        sub_cols = comb_copy
                    results = reg_calculator.init_get_result(x[sub_cols], y)
                    if distinct_cols not in best_frames:
                        best_value = None
                    else:
                        best_value = best_frames[distinct_cols][0]
                    interpret_result = interpret.interpret(results, best_value)
                    if interpret_result is not None:
                        best_frames[distinct_cols] = (interpret_result, sub_cols)

            if verbose and levels < 2:
                progress += step_ratio
                print(f'{progress * 100:.1f} %', end='\r')

    @staticmethod
    def _find_distinct_columns(cols: list[str], count_inter=True) -> list[str]:
        distinct_cols = []
        for col in cols:
            inter_split = col.split('*')
            split_len = len(inter_split)
            if split_len > 1:
                if count_inter:
                    for split_col in inter_split:
                        if not contains(distinct_cols, split_col):
                            distinct_cols.append(split_col)
                continue
            exp_split = col.split('-')
            split_len = len(exp_split)
            if split_len == 2:
                split_col = exp_split[0]
                if not contains(distinct_cols, split_col):
                    distinct_cols.append(split_col)
                continue
            if not contains(distinct_cols, col):
                distinct_cols.append(col)
        return distinct_cols
