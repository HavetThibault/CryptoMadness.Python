import numpy as np

from ml_sdk.analyze.models.models_analyzer import ModelsAnalyzer
from ml_sdk.training.models_results_infos import ModelsResultsInfos
from ml_sdk.training.parameter import Parameter


class ValLossModelsAnalyzer(ModelsAnalyzer):
    def get_metric(self, params_set: list[Parameter], models_results: ModelsResultsInfos) -> float:
        results = models_results.get_training_results(params_set)
        val_losses = [stat.get_val_loss() for stat in results.get_stats()]
        return np.mean(val_losses)