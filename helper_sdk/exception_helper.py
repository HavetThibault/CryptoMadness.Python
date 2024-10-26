import traceback


def format_exception_to_str(e):
    error_message = str(e)
    stack_trace = traceback.format_exc()
    return f"Error: {error_message}\nStack Trace:\n{stack_trace}"