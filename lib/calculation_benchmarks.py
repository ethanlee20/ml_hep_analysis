import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_bin_counts(data, binning_variable, bin_edges):
    bin_series = pd.cut(data[binning_variable], bin_edges, include_lowest=True)
    data_by_bin = data.groupby(bin_series)
    counts = data_by_bin.size()
    return counts


def find_bin_middles(bin_edges):
    """Assumes uniform bin widths"""
    num_bins = len(bin_edges)-1
    bin_width = (np.max(bin_edges) - np.min(bin_edges)) / num_bins
    shifted_edges = bin_edges + 0.5*bin_width
    return shifted_edges[:-1]


def calculate_efficiency(data1, data2, variable, bin_edges):
    bin_counts_data1 = find_bin_counts(data1, variable, bin_edges)
    bin_counts_data2 = find_bin_counts(data2, variable, bin_edges)
    return (bin_counts_data1 / bin_counts_data2).values


