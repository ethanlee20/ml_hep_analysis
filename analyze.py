import os.path
import sys

import pandas as pd

import lib.physics

# Setup
path_input_cut_data = sys.argv[1]
path_output_file = sys.argv[2]

df_B0 = pd.read_pickle(path_input_cut_data)

df_B_4mom = lib.physics.four_momemtum_dataframe(df_B0[["E", "px", "py", "pz"]])

df_mu_p_4mom = lib.physics.four_momemtum_dataframe(df_B0[["mu_p_E", "mu_p_px", "mu_p_py", "mu_p_pz"]])
df_mu_m_4mom = lib.physics.four_momemtum_dataframe(df_B0[["mu_m_E", "mu_m_px", "mu_m_py", "mu_m_pz"]])

df_K_4mom = lib.physics.four_momemtum_dataframe(df_B0[["K_m_E", "K_m_px", "K_m_py", "K_m_pz"]])
df_KST_4mom = lib.physics.four_momemtum_dataframe(df_B0[["KST0_E", "KST0_px", "KST0_py", "KST0_pz"]])

# Solving

df_B0["q_squared"] = lib.physics.invariant_mass_squared_two_particles(df_mu_p_4mom, df_mu_m_4mom)

df_B0["costheta_mu"] = lib.physics.find_costheta_mu(
    df_mu_p_4mom, df_mu_m_4mom, df_B_4mom
)

df_B0["costheta_K"] = lib.physics.find_costheta_K(df_K_4mom, df_KST_4mom, df_B_4mom)

df_B0 = df_B0.rename(columns={"cosHelicityAngle__bo0__cm0__bc": "costheta_K_builtin"})

df_B0["coschi"] = lib.physics.find_coschi(
    df_B_4mom, df_K_4mom, df_KST_4mom, df_mu_p_4mom, df_mu_m_4mom
)

df_B0["chi"] = lib.physics.find_chi(
    df_B_4mom,
    df_K_4mom,
    df_KST_4mom,
    df_mu_p_4mom,
    df_mu_m_4mom,
)

# Saving

df_B0.to_pickle(path_output_file)
