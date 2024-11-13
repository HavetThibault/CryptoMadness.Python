import pandas as pd


class ModelLabelsAndPreds:
    def __init__(self, model_name, param_set, labels_and_preds, training_mem_index, training_mem_sub_index):
        self._model_name: str = model_name
        self._param_set = param_set
        self._training_mem_index: int = training_mem_index
        self._training_mem_sub_index: int = training_mem_sub_index
        self._labels_and_preds: pd.DataFrame = labels_and_preds

    def get_model_name(self):
        return self._model_name

    def get_param_set(self):
        return self._param_set

    def get_labels_and_preds(self):
        return self._labels_and_preds

    def get_training_mem_index(self):
        return self._training_mem_index

    def get_training_mem_sub_index(self):
        return self._training_mem_sub_index
