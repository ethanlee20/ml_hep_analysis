import uproot


# Utilities


def open_root_tree_as_df(path_tree):
    df = uproot.open(path_tree).arrays(library="pd")
    return df
