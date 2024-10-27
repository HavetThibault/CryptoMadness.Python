from ml_sdk.analyse.model_labels_and_preds import ModelLabelsAndPreds


class ClassModelLabelsAndPreds(ModelLabelsAndPreds):
    def __init__(self, model_name, param_set, labels_and_preds, params_accuracy, training_mem_index,
                 training_mem_sub_index):
        super(ClassModelLabelsAndPreds, self).__init__(model_name, param_set, labels_and_preds, training_mem_index,
                                                       training_mem_sub_index)
        self._params_accuracy = params_accuracy

    def get_params_accuracy(self):
        return self._params_accuracy
