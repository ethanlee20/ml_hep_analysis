import lib.physics
import numpy as np


# Cuts


def kst_invmass_cut(df_data):
    def invmass_k_pi(df_data):
        df_k_4mom = df_data[["K_m_E", "K_m_px", "K_m_py", "K_m_pz"]]
        df_pi_4mom = df_data[["pi_p_E", "pi_p_px", "pi_p_py", "pi_p_pz"]]
        df_invass_k_pi = np.sqrt(
            lib.physics.invariant_mass_squared_two_particles(df_k_4mom, df_pi_4mom)
        )
        return df_invass_k_pi

    def cut_on_invM_K_pi(df_data):
        invM_Kst = 0.892
        Kst_full_width = 0.05
        df_invM_K_pi = invmass_k_pi(df_data)
        cut = abs(df_invM_K_pi - invM_Kst) <= 1.5 * Kst_full_width
        df_cut_data = df_data[cut]
        return df_cut_data

    df_data = df_data.copy()
    df_cut_data = cut_on_invM_K_pi(df_data)

    return df_cut_data


def Mbc_cut(df_data):
    df_data = df_data.copy()

    Mbc_low_bound = 5.27
    cut = df_data["Mbc"] > Mbc_low_bound
    df_cut_data = df_data[cut]
    return df_cut_data


def deltaE_cut(df_data):
    df_data = df_data.copy()
    cut = abs(df_data["deltaE"]) <= 0.05
    df_cut_data = df_data[cut]
    return df_cut_data
