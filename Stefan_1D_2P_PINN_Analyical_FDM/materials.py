# materials.py
# ============================================================
# Material parameters for explicit/implicit FDM baseline
# of 1D Stefan laser melting.
#
# Regime 1:
#   Ag, Al, Cu, Ti
#   I = 1e9 W/m²
#   t = t_melt ... 10 s
#
# Regime 2:
#   Ti-6Al-4V
#   Ngwenya-style comparison
#   t = 0 ... 7 μs
#   I = 5, 50, 500, 5000 kW/cm² with I_scale = 1000
# ============================================================


MATERIALS_LONG = {
    "Ag": {
        "rho_s": 10500.0,
        "rho_l": 9300.0,
        "ks": 429.0,
        "kl": 361.0,
        "alpha_s": 1.737506e-04,
        "alpha_l": 1.330940e-04,
        "Tm": 1234.0,
        "T0": 300.0,
        "Lh": 1.112e5,
        "A_s": 0.020,
        "A_l": 0.043,
        "t_melt": 1.15,
        "t_max": 10.0,
        "I_laser": 1e9,
    },

    "Al": {
        "rho_s": 2700.0,
        "rho_l": 2385.0,
        "ks": 238.0,
        "kl": 100.0,
        "alpha_s": 9.613282e-05,
        "alpha_l": 3.880658e-05,
        "Tm": 933.0,
        "T0": 300.0,
        "Lh": 3.880e5,
        "A_s": 0.0588,
        "A_l": 0.064,
        "t_melt": 0.034,
        "t_max": 10.0,
        "I_laser": 1e9,
    },

    "Cu": {
        "rho_s": 8960.0,
        "rho_l": 8000.0,
        "ks": 401.0,
        "kl": 342.0,
        "alpha_s": 1.161117e-04,
        "alpha_l": 8.906250e-05,
        "Tm": 1358.0,
        "T0": 300.0,
        "Lh": 2.047e5,
        "A_s": 0.020,
        "A_l": 0.058,
        "t_melt": 1.94,
        "t_max": 10.0,
        "I_laser": 1e9,
    },

    "Ti": {
        "rho_s": 4500.0,
        "rho_l": 4110.0,
        "ks": 21.6,
        "kl": 20.28,
        "alpha_s": 9.090909e-06,
        "alpha_l": 7.054745e-06,
        "Tm": 1940.0,
        "T0": 300.0,
        "Lh": 3.650e5,
        "A_s": 0.257,
        "A_l": 0.433,
        "t_melt": 1.045e-3,
        "t_max": 10.0,
        "I_laser": 1e9,
    },
}


TI64 = {
    "rho_s": 4510.0,
    "rho_l": 4510.0,
    "ks": 20.0,
    "kl": 29.0,
    "alpha_s": 5.8e-6,
    "alpha_l": 5.95e-6,
    "Tm": 1928.0,
    "T0": 300.0,
    "Lh": 2.9e5,
    "A_s": 0.433,
    "A_l": 0.433,
    "t_melt": 0.0,
    "t_max": 7e-6,
}


TI64_INTENSITIES_KW_CM2 = [5, 50, 500, 5000]


def ti64_effective_intensity_W_m2(I_kW_cm2, I_scale=1000.0):
    """
    Convert kW/cm² to effective W/m².

    1 kW/cm² = 1e3 W/cm² = 1e7 W/m²

    In the user's Ngwenya-style implementation:
        AI_eff = A * I_W_m2 * I_scale

    Therefore this function returns:
        I_eff = I_W_m2 * I_scale
    """
    I_W_m2 = float(I_kW_cm2) * 1e7
    return I_W_m2 * float(I_scale)