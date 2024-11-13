from ml_sdk.analyze.model_labels_and_preds import ModelLabelsAndPreds
from ml_sdk.analyze.model_labels_and_preds_instantiator import ModelLabelsAndPredsInstantiator
from ml_sdk.analyze.predictions_metrics import calculate_labels_and_predictions


class RegModelLabelsAndPredsInstantiator(ModelLabelsAndPredsInstantiator):
    def instantiate(self, model, val_ds, val_len, batch_size, model_name, params_set, progress,
                    training_mem_index, training_mem_sub_index) -> ModelLabelsAndPreds:
        labels_and_preds = calculate_labels_and_predictions(model, val_ds, batch_size, val_len, progress)
        return ModelLabelsAndPreds(
            model_name,
            params_set,
            labels_and_preds,
            training_mem_index,
            training_mem_sub_index)
