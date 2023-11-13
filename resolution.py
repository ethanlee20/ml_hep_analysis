import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import lib.plotting

lib.plotting.setup_mpl_params_save()


data_generator = pd.read_pickle('~/Desktop/data/2023-11-10_10k_mu_events/analyzed_cut_generator_mu.pkl')
data_generator_med_q2 = data_generator[(data_generator['q_squared'] < 6) & (data_generator['q_squared'] > 1)]
data_detector = pd.read_pickle('~/Desktop/data/2023-11-10_10k_mu_events/analyzed_cut_detector_mu.pkl')
data_detector_med_q2 = data_detector[(data_detector['q_squared'] < 6) & (data_detector['q_squared'] > 1)]





