import math
from typing import Any

import keras
import numpy as np
import pandas as pd

from helper_sdk.list_helper import str_list_equals
from helper_sdk.work_progress_state import WorkProgressState
from ml_sdk.dataset.float_categories_calculator import FloatCategoriesCalculator


# 'ds' batch size must be 1 !
def calculate_labels_and_predictions(model: keras.Model, ds, batch_size: int, ds_len: int, progress: WorkProgressState = None) \
        -> pd.DataFrame:
    if progress is not None:
        progress.start_resume()
    ds_iter = iter(ds)
    df_rows = []
    output_names = []
    for i in range(int(ds_len / batch_size) + 1):
        ds_batch = next(ds_iter)
        input_batch: dict[str, Any] = ds_batch[0]
        predictions: dict[str, np.ndarray] = model.predict(input_batch, verbose=False)
        labels: dict[str, Any] = ds_batch[1]
        if i == 0:
            if not str_list_equals(list(predictions.keys()), list(labels.keys())):
                raise Exception('The predictions of the models :\n'
                                f'{list(predictions.keys())}'
                                f'Must match the dataset headers:\n'
                                f'{list(labels.keys())}')
            output_names = list(predictions.keys())
        if not i == int(ds_len / batch_size):
            to_do_rows = batch_size
        else:
            to_do_rows = ds_len % batch_size
        for k in range(to_do_rows):
            df_row = []
            for output_name in output_names:
                label: np.ndarray = labels[output_name].numpy()[k]
                prediction = predictions[output_name][k]
                if label.shape[0] > 1:
                    for z, i_label in enumerate(label):
                        df_row.append(i_label)
                        df_row.append(prediction[z])
                else:
                    df_row.append(label[0])
                    df_row.append(prediction[0])
            df_rows.append(df_row)
            if progress is not None:
                progress.increment_done()
    df_cols = []
    for output_name in output_names:
        label: np.ndarray = labels[output_name].numpy()[0]
        if label.shape[0] > 1:
            for i in range(len(label)):
                df_cols.append(output_name + f'_{i}')
                df_cols.append(output_name + f'_{i} prediction')
        else:
            df_cols.append(output_name)
            df_cols.append(output_name + ' prediction')
    return pd.DataFrame(df_rows, columns=df_cols)


# Returns a list of rows. Each row containing one list of tuple 'isRightClass?' - 'model_certainty' per label category
def calculate_classification_labels_and_predictions(model, ds, ds_len, batch_size,
                                                    progress: WorkProgressState = None) \
        -> list[list[list[tuple[int, float]]]]:
    if progress is not None:
        progress.start_resume()
    ds_iter = iter(ds)
    labels_and_predictions = []
    for i in range(int(ds_len / batch_size) + 1):
        ds_batch = next(ds_iter)
        input_batch = ds_batch[0]
        labels = ds_batch[1]
        # The model.predict returns one list per category estimation. Each of these lists is a list of <batch_size> list
        # It is the same for the variable 'labels'
        prediction_batch = model.predict(input_batch, verbose=False)
        for k in range(batch_size):
            output_row = []
            for p in range(len(prediction_batch)):
                output_category_row = []
                labels_category_iter = iter(labels[p][k])
                for prediction in prediction_batch[p][k]:
                    output_category_row.append((next(labels_category_iter), prediction))
                output_row.append(output_category_row)
            labels_and_predictions.append(output_row)
            if progress is not None:
                progress.increment_done()
    return labels_and_predictions


# Return the index of the predicted class, the certainty of the prediction, the index of the real class
def get_predicted_class_and_co(prediction: list[tuple[int, float]]) -> tuple[int, float, int]:
    real_class_index = None
    predicted_index = None
    prediction_certainty = 0
    for i, (real_class, prediction) in enumerate(prediction):
        if real_class == 1:
            if real_class_index is not None:
                raise Exception('An image can only belong to one class. Must be a code mistake.')
            real_class_index = i
        if prediction > prediction_certainty:
            prediction_certainty = prediction
            predicted_index = i
    if real_class_index is None:
        raise Exception('Column with one hot encoding label was not loaded as a float or there is a code mistake !')
    return predicted_index, prediction_certainty, real_class_index


def calculate_classification_predictions_and_accuracy(
        raw_predictions: list[list[list[tuple[int, float]]]],
        categories_calculators: list[FloatCategoriesCalculator]) \
            -> tuple[list[list[tuple[float, float]]], list[float]]:
    classification_predictions = []
    predictions = 0
    correct_predictions = [0 for _ in range(len(categories_calculators))]
    for raw_prediction in raw_predictions:
        classification_prediction = []
        for i, category_prediction in enumerate(raw_prediction):
            categories_calculator = categories_calculators[i]
            predicted_index, prediction_certainty, real_class_index = get_predicted_class_and_co(category_prediction)
            predicted = categories_calculator.get_category_center(predicted_index)
            real_class = categories_calculator.get_category_center(real_class_index)
            classification_prediction.append((real_class, predicted))
            if predicted_index == real_class_index:
                correct_predictions[i] += 1
        predictions += 1
        classification_predictions.append(classification_prediction)
    for i in range(len(correct_predictions)):
        correct_predictions[i] /= predictions
    return classification_predictions, correct_predictions


def calculate_img_labels_and_predictions(model, img_names, ds, ds_len, batch_size) \
        -> dict[str, list[tuple[float, float]]]:
    ds_iter = iter(ds)
    img_iter = iter(img_names)
    labels_and_predictions = {}
    for i in range(int(ds_len / batch_size) + 1):
        ds_batch = next(ds_iter)
        prediction_batch = model.predict(ds_batch[0], verbose=False)
        label_batch = ds_batch[1].numpy()
        for p, predictions in enumerate(prediction_batch):
            img_name = next(img_iter, None)
            if img_name is None:
                break
            label_and_prediction = []
            labels = label_batch[p]
            for z, prediction_value in enumerate(predictions):
                label_and_prediction.append((labels[z], prediction_value))

            labels_and_predictions[img_name] = label_and_prediction
    return labels_and_predictions


def get_per_param_abs_errors_mean_std(labels_and_predictions: list[list[tuple[float, float]]]) \
        -> list[tuple[float, float]]:
    all_errors = []
    for i in range(len(labels_and_predictions[0])):
        all_errors.append([])

    for label_and_prediction in labels_and_predictions:
        for k, (label, prediction) in enumerate(label_and_prediction):
            all_errors[k].append(label - prediction)

    error_mean_std = []
    for errors in all_errors:
        error_mean_std.append((float(np.mean(np.abs(errors))), float(np.std(errors))))

    return error_mean_std


def get_errors_mean_square(labels_and_predictions: list[list[tuple[float, float]]]) -> float:
    error = 0
    params_nbr = len(labels_and_predictions[0])
    for label_and_prediction in labels_and_predictions:
        example_error = 0
        for k, (label, prediction) in enumerate(label_and_prediction):
            example_error += (label - prediction) * (label - prediction)
        error += example_error / params_nbr
    return error / len(labels_and_predictions)


def get_error_mean_abs_percentage(labels_and_predictions: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    all_errors = []
    for i in range(len(labels_and_predictions[0])):
        all_errors.append([])

    for label_and_prediction in labels_and_predictions:
        for i, (label, prediction) in enumerate(label_and_prediction):
            all_errors[i].append(math.fabs((label - prediction)/label))

    error_mean_std = []
    for errors in all_errors:
        error_mean_std.append((float(np.mean(errors)), float(np.std(errors))))

    return error_mean_std


def get_params_intervals(ds: pd.DataFrame, start_col=0) -> list[tuple[float, float]]:
    intervals = []
    for i in range(start_col, len(ds.columns)):
        col = ds[[ds.columns[i]]]
        intervals.append((float(col.min(axis=0)), float(col.max(axis=0))))
    return intervals
