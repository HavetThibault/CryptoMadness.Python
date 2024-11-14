import datetime

from helper_sdk.datetime_helper import format_datetime_round_s, format_delta_round_s


def default_progress_display(done, total, left_time):
      print(f'{done}/{total}\n'
            f'{format_datetime_round_s(datetime.datetime.now())}\n'
            f'Left time: {format_delta_round_s(left_time)}\n')


def get_default_progress_state(total: int = None):
    progress = WorkProgressState(total, 1)
    progress.add_listener(default_progress_display)
    return progress


class WorkProgressState:
    def __init__(self, total = None, estimation_step = 1):
        self._estimation_step = estimation_step

        self._done = 0
        self._total = total

        self._next_estimation = self._estimation_step
        self._task_start = None
        self._listeners = []
        self._total_time = datetime.timedelta()

    def reset(self, total, estimation_step):
        self._estimation_step = estimation_step
        self._done = 0
        self._total = total

        self._next_estimation = self._estimation_step
        self._task_start = None
        self._total_time = datetime.timedelta()

    def get_savable(self):
        return (
            self._estimation_step,
            self._done,
            self._total,
            self._next_estimation,
            self._total_time
        )

    @staticmethod
    def from_saved(saved):
        (
            estimation_step,
            done,
            total,
            next_estimation,
            total_time
        ) = saved
        progress = WorkProgressState(total, estimation_step)
        progress._done = done
        progress._next_estimation = next_estimation
        progress._total_time = total_time
        return progress

    def get_total(self):
        return self._total

    def get_done(self):
        return self._done

    def get_left_time(self):
        if self._done > 0:
            return self._total_time/self._done * (self._total - self._done)
        return None

    def add_listener(self, listener):
        self._listeners.append(listener)

    def remove_listener(self, listener):
        self._listeners.remove(listener)

    def _call_listeners(self):
        for listener in self._listeners:
            listener(self._done, self._total, self.get_left_time())

    def start_resume(self):
        self._task_start = datetime.datetime.now()
        if self._done == 0:
            self._call_listeners()

    def increment_done(self):
        self._done += 1
        if self._done >= self._next_estimation or self._done == self._total:
            now = datetime.datetime.now()
            task_time = now - self._task_start
            self._total_time += task_time

            self._call_listeners()

            self._next_estimation += self._estimation_step
            self._task_start = now
