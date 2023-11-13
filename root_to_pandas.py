import sys

import lib.util

path_input_tree = sys.argv[1]
path_output_pickle = sys.argv[2]
 
df = lib.util.open_root_tree_as_df(path_input_tree)
df.to_pickle(path_output_pickle)
