from ml_sdk.training.models_results_infos import ModelsResultsInfos
from ml_sdk.training.parameter import Parameter


class ModelsAnalyzer:
    # Returns the metrics of th model. The bigger the metric, the better the model
    def get_metric(self, params_set: list[Parameter], results: ModelsResultsInfos) -> float:
        raise NotImplementedError()
