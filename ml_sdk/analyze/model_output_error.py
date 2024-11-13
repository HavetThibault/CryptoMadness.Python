import pandas as pd


class ModelsOutputsErrors:
    def __init__(self, error_name):
        self._error_name: str = error_name
        self._models_errors = []

    def get_error(self):
        return self._error_name

    def add_model_error(self, model_name, param_set, err_mean, err_std):
        self._models_errors.append((model_name, param_set, err_mean, err_std))

    def get_df(self):
        self._models_errors.sort(key=self._get_err_mean)
        return pd.DataFrame(self._models_errors, columns=['Model name', 'Parameters set', 'Error mean', 'Error std'])

    @staticmethod
    def _get_err_mean(row):
        return row[2]
