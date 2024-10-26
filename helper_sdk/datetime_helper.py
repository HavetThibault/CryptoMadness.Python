import datetime


def format_delta_round_s(delta: datetime.timedelta) -> str:
    delta_str = str(delta)
    split = delta_str.split(':')
    if split[-1].__contains__('.'):
        ms_len = len(split[-1].split('.')[-1])
        return delta_str[:-ms_len-1]
    return delta_str


def format_datetime_round_s(date: datetime.datetime) -> str:
    date_str = str(date)
    if date_str.__contains__('.'):
        ms_len = len(date_str.split('.')[-1])
        return date_str[:-ms_len - 1]
    return date_str

