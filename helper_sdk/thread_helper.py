import ctypes


# exception is the class itself. Example: 'SystemExit'
def raise_thread_exception(thread_id, exception_type=SystemExit):
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(exception_type))

    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res > 1:
        # Reverting the exception as we might have raised it in more than one thread
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")