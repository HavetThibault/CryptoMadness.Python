from model.training.analyse.model_labels_and_preds import ModelLabelsAndPreds


class ModelLabelsAndPredsInstantiator:
    def instantiate(self, model, val_ds, val_len, batch_size, model_name, params_set, progress,
                    training_mem_index, training_mem_sub_index) -> ModelLabelsAndPreds:
        raise Exception('Method "instantiate" not implemented')
