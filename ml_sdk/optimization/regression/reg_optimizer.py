import pandas as pd

from ml_sdk.optimization.parameter_interval import ParameterInterval
from ml_sdk.optimization.regression.ds_transformer import DSTransformer
from ml_sdk.optimization.regression.reg_calculator import RegCalculator
from ml_sdk.optimization.regression.reg_helper import first_significant_best_frame
from ml_sdk.optimization.regression.reg_results_interpret import RegResultsInterpret
from ml_sdk.training.error_calculator import ErrorCalculator
from ml_sdk.training.trainings_memory import TrainingsMemory
from ml_sdk.training.trainings_runner import lowest_error_params_set


def optimize_and_print(
        params_intervals: list[ParameterInterval],
        cols: list[str],
        training_mem_path,
        error_calc: ErrorCalculator,
        reg_calc: RegCalculator,
        ds_transformer: DSTransformer,
        results_interpret: RegResultsInterpret,
        dir_prefix: str,
        verbose: bool):
    training_mem: TrainingsMemory = TrainingsMemory.load_instance(training_mem_path)
    independent_var = training_mem.get_params_df(cols)
    dependent_var = training_mem.get_target_df(error_calc)
    print('======================== Loaded ========================', independent_var.join(dependent_var), sep='\n')
    x = pd.DataFrame(independent_var, columns=cols)
    poly_x = ds_transformer.transform(x)

    best_frame = first_significant_best_frame(
        x,
        dependent_var,
        ds_transformer,
        reg_calc,
        cols,
        results_interpret,
        verbose
    )
    reg_calc.init_model_and_result(poly_x[best_frame], dependent_var)
    params_set, error = lowest_error_params_set(
        best_frame,
        params_intervals,
        dir_prefix,
        cols,
        ds_transformer,
        reg_calc)

    print('======================== Best regression model stats ========================', reg_calc.reg_results.summary(), sep='\n')
    print(f'\n============ Best frame: {best_frame} ============')
    print('Best params set:', params_set)
    print('==> Error:', error)
