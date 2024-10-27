import numpy as np


def display_hist(values, intervals: int):
    intervals_edges = []
    min = np.min(values)
    max = np.max(values)
    interval_len = (max - min + 0.05) / intervals
    edge = min - 0.025
    for i in range(intervals):
        intervals_edges.append((edge*2 + interval_len)/2)
