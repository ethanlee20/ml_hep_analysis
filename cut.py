import os.path
import sys

import pandas as pd

import lib.cuts

path_input_data = sys.argv[1]
path_output_file = sys.argv[2]

df = pd.read_pickle(path_input_data)

df_cut1 = lib.cuts.kst_invmass_cut(df)
df_cut2 = lib.cuts.Mbc_cut(df_cut1)
df_cut3 = lib.cuts.deltaE_cut(df_cut2)

df_cut3.to_pickle(path_output_file)
