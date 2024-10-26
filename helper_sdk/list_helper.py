import math


def contains(src: list, value) -> bool:
    for current_value in src:
        if current_value == value:
            return True
    return False


def float_list_equals(list1: list[float], list2: list[float]) -> bool:
    for k in range(len(list1)):
        if not math.isclose(list1[k], list2[k]):
            return False
    return True


def str_list_equals(list1: list[str], list2: list[str]) -> bool:
    if len(list1) != len(list2):
        return False
    for value in list1:
        if not contains(list2, value):
            return False
    return True