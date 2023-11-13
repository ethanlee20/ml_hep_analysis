import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import lib.plotting

lib.plotting.setup_mpl_params_save()


data_generator = pd.read_pickle('~/Desktop/data/2023-11-10_10k_mu_events/analyzed_cut_generator_mu.pkl')
data_generator_med_q2 = data_generator[(data_generator['q_squared'] < 6) & (data_generator['q_squared'] > 1)]
data_detector = pd.read_pickle('~/Desktop/data/2023-11-10_10k_mu_events/analyzed_cut_detector_mu.pkl')
data_detector_med_q2 = data_detector[(data_detector['q_squared'] < 6) & (data_detector['q_squared'] > 1)]


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

def plot_efficiency(data_recon, data_gen, variable, bin_edges, title, xlabel):
    efficiency = calculate_efficiency(data_recon, data_gen, variable, bin_edges)
    bin_middles = find_bin_middles(bin_edges)
    fig, ax = plt.subplots()
    ax.scatter(
        bin_middles,
        efficiency,
        label=f'Reconstructed events: {len(data_recon)}\nGenerator events: {len(data_generator)}',
        color='red'
    )
    ax.legend()
    ax.set_ylim(0,1)
    ax.set_ylabel(r'$\varepsilon$', rotation=0, labelpad=20)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

def generate_bin_edges(start, stop, num_of_bins):
    bin_size = (stop-start) / num_of_bins
    return np.arange(start, stop+bin_size, bin_size)



bin_edges_costheta_K = generate_bin_edges(0,1,10)

plot_efficiency(data_detector, data_generator, 'costheta_K', bin_edges_costheta_K, r'Efficiency of $\cos\theta_K$', r'$\cos\theta_K$')
plt.savefig('eff_costheta_K.png', bbox_inches='tight')
plt.close()



bin_edges_costheta_mu = generate_bin_edges(0,1,10)

plot_efficiency(data_detector, data_generator, 'costheta_mu', bin_edges_costheta_mu, r'Efficiency of $\cos\theta_\mu$', r'$\cos\theta_\mu$')
plt.savefig('eff_costheta_mu.png', bbox_inches='tight')
plt.close()




bin_edges_chi = generate_bin_edges(0, 2*np.pi,10)

plot_efficiency(data_detector, data_generator, 'chi', bin_edges_chi, r'Efficiency of $\chi$', r'$\chi$')
plt.savefig('eff_chi.png', bbox_inches='tight')
plt.close()















bin_edges_costheta_K = generate_bin_edges(0,1,10)

plot_efficiency(data_detector_med_q2, data_generator_med_q2, 'costheta_K', bin_edges_costheta_K, r'Efficiency of $\cos\theta_K$', r'$\cos\theta_K$')
plt.savefig('med_q2_eff_costheta_K.png', bbox_inches='tight')
plt.close()



bin_edges_costheta_mu = generate_bin_edges(0,1,10)

plot_efficiency(data_detector_med_q2, data_generator_med_q2, 'costheta_mu', bin_edges_costheta_mu, r'Efficiency of $\cos\theta_\mu$', r'$\cos\theta_\mu$')
plt.savefig('med_q2_eff_costheta_mu.png', bbox_inches='tight')
plt.close()




bin_edges_chi = generate_bin_edges(0,2*np.pi,10)

plot_efficiency(data_detector_med_q2, data_generator_med_q2, 'chi', bin_edges_chi, r'Efficiency of $\chi$', r'$\chi$')
plt.savefig('med_q2_eff_chi.png', bbox_inches='tight')
plt.close()



## plot Lucas plot

## all q2
bins= 20
fig, ax = plt.subplots()
ax.hist(
    data_generator['costheta_K'],
    label=f'Generator (Events: {len(data_generator)})',
    color='purple',
    bins=bins,
    histtype='step',
    linestyle='-',
    #linewidth=1.3
)
ax.hist(
    data_detector['costheta_K'],
    label=f'Reconstructed (Events: {len(data_detector)})',
    color='blue',
    bins=bins,
    histtype='step'
)
ax.hist(
    data_detector['costheta_K_builtin'],
    label='Reconstructed basf2',
    color='red',
    bins=bins,
    histtype='step',
    linestyle='--',
    linewidth=1.2
)
ax.set_title(r'Comparison of $\cos\theta_K$')
ax.set_xlabel(r'$\cos\theta_K$')

ax.legend()

plt.savefig('comparison.png', bbox_inches='tight')


## med q2

bins= 20
fig, ax = plt.subplots()
ax.hist(
    data_generator_med_q2['costheta_K'],
    label=f'Generator (Events: {len(data_generator_med_q2)})',
    color='purple',
    bins=bins,
    histtype='step',
    linestyle='-',
    #linewidth=1.3
)
ax.hist(
    data_detector_med_q2['costheta_K'],
    label=f'Reconstructed (Events: {len(data_detector_med_q2)})',
    color='blue',
    bins=bins,
    histtype='step'
)
ax.hist(
    data_detector_med_q2['costheta_K_builtin'],
    label='Reconstructed basf2',
    color='red',
    bins=bins,
    histtype='step',
    linestyle='--',
    linewidth=1.2
)
ax.set_title(r'Comparison of $\cos\theta_K$')
ax.set_xlabel(r'$\cos\theta_K$')

ax.legend()

plt.savefig('med_q2_comparison.png', bbox_inches='tight')