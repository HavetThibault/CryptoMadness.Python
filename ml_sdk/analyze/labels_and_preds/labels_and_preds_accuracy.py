import math

import pandas as pd

from helper_sdk.csv_helper import get_csv_writer
from ml_sdk.analyze.classification.output_classes import OutputClasses
from ml_sdk.analyze.classification.specific_class_metrics import SpecificClassMetrics
from ml_sdk.analyze.labels_and_preds.labels_and_preds_processor import LabelsAndPredsProcessor
from ml_sdk.analyze.predictions_metrics import PREDICTION_COL_END
from ml_sdk.analyze.success_rate import SuccessRate


class LabelsAndPredsAcccuracy(LabelsAndPredsProcessor):
    def __init__(self, filepath: str, intervals: list[tuple[float, float]], classes: list[OutputClasses]):
        self._filepath = filepath
        self._intervals = intervals
        self._classes: list[OutputClasses] = classes

    def process_start(self):
        pass

    def process_end(self):
        pass

    def process(self, params_set, filename, predictions: pd.DataFrame):
        sub_classes_metrics = dict[str, SpecificClassMetrics]()
        classes_metrics = dict[str, SuccessRate]()
        for output_class in self._classes:
            classes_metrics[output_class.get_main()] = SuccessRate()
            for specific_class in output_class.get_classes():
                sub_classes_metrics[specific_class] = SpecificClassMetrics(len(self._intervals))

        rows_iter = iter(predictions)
        for row in rows_iter:
            for output_class in self._classes:
                max_class = None
                max_class_value = None
                for specific_class in output_class.get_classes():
                    specific_class_value = row[specific_class]
                    if max_class_value is None or specific_class_value > max_class_value:
                        max_class_value = specific_class_value
                        max_class = specific_class
                # According to 'calculate_labels_and_predictions' in predictions_metrics
                if math.isclose(1, row[max_class + PREDICTION_COL_END], rel_tol=0.0001):
                    sub_classes_metrics[max_class].add_right(self._get_interval_index(max_class_value))
                    classes_metrics[output_class.get_main()].add_right()
                else:
                    sub_classes_metrics[max_class].add_wrong(self._get_interval_index(max_class_value))
                    classes_metrics[output_class.get_main()].add_wrong()

        with open(self._filepath, 'w', newline='\n') as csvfile:
            csv_writer = get_csv_writer(csvfile)
            for output_class in self._classes:
                for specific_class in output_class.get_classes():
                    csv_writer.writerow(sub_classes_metrics[specific_class].get_rates())

            csv_writer.writerow([''])
            for output_class in self._classes:
                csv_writer.writerow([classes_metrics[output_class.get_main()].get_rate()])

    def _get_interval_index(self, value: float) -> int:
        for (i, (interval_min, interval_max)) in enumerate(self._intervals):
            if value <= interval_max:
                return i
