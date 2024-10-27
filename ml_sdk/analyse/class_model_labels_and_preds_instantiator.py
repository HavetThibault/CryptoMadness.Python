from ml_sdk.analyse.class_model_labels_and_preds import ClassModelLabelsAndPreds
from ml_sdk.analyse.model_labels_and_preds import ModelLabelsAndPreds
from ml_sdk.analyse.model_labels_and_preds_instantiator import ModelLabelsAndPredsInstantiator
from ml_sdk.analyse.predictions_metrics import calculate_classification_labels_and_predictions, \
    calculate_classification_predictions_and_accuracy
from ml_sdk.dataset.float_categories_calculator import FloatCategoriesCalculator


class ClassModelLabelsAndPredsInstantiator(ModelLabelsAndPredsInstantiator):
    def __init__(self, categories_calculators: list[FloatCategoriesCalculator]):
        self._categories_calculators = categories_calculators

    def instantiate(self, model, val_ds, val_len, batch_size, model_name, params_set, progress,
                    training_mem_index, training_mem_sub_index) -> ModelLabelsAndPreds:
        labels_and_predictions = calculate_classification_labels_and_predictions(
            model, val_ds, val_len, batch_size, progress)
        labels_and_preds, params_accuracy = (
            calculate_classification_predictions_and_accuracy(
                labels_and_predictions,
                self._categories_calculators))
        return ClassModelLabelsAndPreds(
            model_name,
            params_set,
            labels_and_preds,
            params_accuracy,
            training_mem_index,
            training_mem_sub_index)
