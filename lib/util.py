import uproot
import numpy as np

# Utilities


def open_root_tree_as_df(path_tree):
    df = uproot.open(path_tree).arrays(library="pd")
    return df


def generate_bin_edges(start, stop, num_of_bins):
    bin_size = (stop-start) / num_of_bins
    return np.arange(start, stop+bin_size, bin_size)