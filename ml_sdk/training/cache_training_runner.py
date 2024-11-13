from ml_sdk.analyze.models.models_analyzer import ModelsAnalyzer
from ml_sdk.model.creator.model_creator import ModelCreator
from ml_sdk.optimization.parameters_matrix_generator import params_set_to_str
from ml_sdk.training.best_model_checkpoint import BestModelCheckpoint
from ml_sdk.training.models_results_infos import ModelsResultsInfos, archive
from ml_sdk.training.parameter import Parameter


class CacheTrainingRunner:
    _VERBOSE = 1
    _WEIGHTS_ONLY = True
    _ON_CPU = False

    def __init__(self, model_creator, models_results_infos, model_analyzer, repeat, iterations, add_callbacks,
                 loss, optimizer, error_calc):
        self._model_creator: ModelCreator = model_creator
        self._models_results_infos: ModelsResultsInfos = models_results_infos
        self._model_analyzer: ModelsAnalyzer = model_analyzer
        self._repeat = repeat
        self._iterations = iterations
        self._add_callbacks = add_callbacks
        self._loss = loss
        self._optimizer = optimizer
        self._error_calc = error_calc

    def get_training_metric(self, raw_params_set: list):
        params_set = [Parameter(f'h{i}', value) for i, value in enumerate(raw_params_set)]
        results = self._models_results_infos.get_training_results(params_set)
        weight_dir = self._models_results_infos.get_weights_dir()
        params_set_index = self._models_results_infos.get_params_set_index(params_set)
        while results is None or results.get_stats_count() < self._repeat:
            if results is None:
                sub_params_set_index = 0
            else:
                sub_params_set_index = results.get_stats_count()
            serializer = BestModelCheckpoint(
                weight_dir, self._model_creator.get_model_name(), self._error_calc, self._VERBOSE,
                weights_only=self._WEIGHTS_ONLY)
            print(f'{params_set_index}.{sub_params_set_index}. Training for: {params_set_to_str(params_set)}')
            training_hist = self._model_creator.create_and_train_model(
                params_set,
                self._iterations,
                [serializer] + self._add_callbacks,
                give_up_cpu=True,
                optimizer=self._optimizer,
                verbose=self._VERBOSE,
                loss=self._loss,
                on_cpu=self._ON_CPU)

            archive(self._models_results_infos, params_set, self._model_creator, serializer, training_hist,
                    sub_params_set_index, self._WEIGHTS_ONLY)

            if results is None:
                results = self._models_results_infos.get_training_results(params_set)
        metric = self._model_analyzer.get_metric(params_set, self._models_results_infos)
        print(f'Result for param_set {params_set_to_str(params_set)} is metric: {metric}')
        return metric

