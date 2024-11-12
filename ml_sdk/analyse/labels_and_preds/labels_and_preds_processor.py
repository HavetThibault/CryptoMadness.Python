import pandas as pd


class LabelsAndPredsProcessor:

    def process_start(self):
        raise NotImplementedError()

    def process_end(self):
        raise NotImplementedError()

    def process(self, params_set, filename, predictions: pd.DataFrame):
        raise NotImplementedError()
