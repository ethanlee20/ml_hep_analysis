import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import lib.plotting


lib.plotting.setup_mpl_params_save()


path_input_data_gen_level = sys.argv[1]
path_input_data_recon_level = sys.argv[2]
path_output_dir = sys.argv[3]


data_gen_level = pd.read_pickle(path_input_data_gen_level)
split_data_gen_level = dict(
    q2_all=data_gen_level,
    q2_med=data_gen_level[(data_gen_level['q_squared'] > 1) & (data_gen_level['q_squared'] < 6)]
)
data_recon_level = pd.read_pickle(path_input_data_recon_level)
split_data_recon_level = dict(
    q2_all=data_recon_level,
    q2_med=data_recon_level[(data_recon_level['q_squared'] > 1) & (data_recon_level['q_squared'] < 6)]
)


def plot_efficiencies():

    num_data_points = 10

    variables = ['costheta_K', 'costheta_mu', 'chi']
    titles = dict(
        zip(
            variables,
            [r'Efficiency of $\cos\theta_K$', r'Efficiency of $\cos\theta_\mu$', r'Efficiency of $\chi$']
        )
    )
    xlabels = dict(
        zip(
            variables,
            [r'$\cos\theta_K$', r'$\cos\theta_\mu$', r'$\chi$']
        )
    )

    for split in split_data_recon_level:
        for variable in variables:
            lib.plotting.plot_efficiency(
                data_recon=split_data_recon_level[split],
                data_gen=split_data_gen_level[split],
                variable=variable,
                num_data_points=num_data_points,
                title=titles[variable],
                xlabel=xlabels[variable]
            )
            plt.savefig(os.path.join(path_output_dir, f'{split}_eff_{variable}.png'), bbox_inches='tight')
            plt.close()


def plot_comparison_costheta_K():
    bins = 20

    for split in split_data_recon_level:
        fig, ax = plt.subplots()
        ax.hist(
            split_data_gen_level[split]['costheta_K'],
            label=f'Generator (Events: {len(split_data_gen_level[split])})',
            color='purple',
            bins=bins,
            histtype='step',
            linestyle='-',
        )
        ax.hist(
            split_data_recon_level[split]['costheta_K'],
            label=f'Reconstructed (Events: {len(split_data_gen_level[split])})',
            color='blue',
            bins=bins,
            histtype='step'
        )
        ax.hist(
            split_data_recon_level[split]['costheta_K_builtin'],
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

        plt.savefig(os.path.join(path_output_dir, f'{split}_comparison_costheta_K.png'), bbox_inches='tight')


plot_efficiencies()
plot_comparison_costheta_K()