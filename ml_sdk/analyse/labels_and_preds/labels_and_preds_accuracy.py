import pandas as pd

from ml_sdk.analyse.classification.specific_class_metrics import SpecificClassMetrics
from ml_sdk.analyse.classification.success_rate import SuccessRate
from ml_sdk.analyse.labels_and_preds.labels_and_preds_processor import LabelsAndPredsProcessor
from ml_sdk.analyse.classification.output_classes import OutputClasses


class LabelsAndPredsAcccuracy(LabelsAndPredsProcessor):
    def __init__(self, filepath: str, intervals: list[tuple[float, float]], classes: list[OutputClasses]):
        self._filepath = filepath
        self._intervals = intervals
        self._classes: list[OutputClasses] = classes

    def process(self, df: pd.DataFrame):
        sub_classes_metrics = dict[str, SpecificClassMetrics]()
        classes_metrics = dict[str, SuccessRate]()
        for output_class in self._classes:
            classes_metrics[output_class.get_main()] = SuccessRate()
            for specific_class in output_class.get_classes():
                sub_classes_metrics[specific_class] = SpecificClassMetrics(len(self._intervals))

        rows_iter = iter(df)
        for row in rows_iter:
            for output_class in self._classes:
                classes = [row[specific_class] for specific_class in output_class.get_classes()]

    def _get_interval_index(self, value: float) -> int:
        for (i, (interval_min, interval_max)) in enumerate(self._intervals):
            if value <= interval_max:
                return i
