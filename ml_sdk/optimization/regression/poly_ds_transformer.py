import numpy as np
import pandas as pd

from ml_sdk.optimization.regression.ds_transformer import DSTransformer
from ml_sdk.optimization.regression.reg_frame_finder import CONST_COL


class PolyDSTransformer(DSTransformer):
    def __init__(self, degree, interactions):
        self._degree = degree
        self._interactions = interactions

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        poly_x = None
        for col in x.columns:
            for i in range(1, self._degree + 1):
                if poly_x is None:
                    poly_x = pd.DataFrame({col: x[col]})
                elif i == 1:
                    poly_x = poly_x.join(pd.DataFrame({col: x[col]}))
                else:
                    poly_x = poly_x.join(pd.DataFrame({col + '-' + str(i): x[col] ** i}))
        if self._interactions:
            ncol = len(x.columns)
            for i in range(ncol):
                col1 = x.columns[i]
                for k in range(i + 1, ncol):
                    col2 = x.columns[k]
                    poly_x = poly_x.join(pd.DataFrame({col1 + '*' + col2: x[col1] * x[col2]}))
        return poly_x.join(pd.DataFrame({CONST_COL: np.ones(len(x))}))
