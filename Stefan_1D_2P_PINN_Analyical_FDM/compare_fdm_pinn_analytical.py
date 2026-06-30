# compare_fdm_pinn_analytical.py
# ============================================================
# Run explicit and implicit FDM baselines for:
#
# 1) Ag, Al, Cu, Ti:
#    t = t_melt ... 10 s
#    unit = cm
#
# 2) Ti-6Al-4V:
#    t = 0 ... 7 μs
#    unit = μm
#
# Saves:
#   plots/
#   results/
# ============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from materials import (
    MATERIALS_LONG,
    TI64,
    TI64_INTENSITIES_KW_CM2,
    ti64_effective_intensity_W_m2,
)

from fdm_solver import (
    solve_fdm_enthalpy_explicit,
    solve_fdm_enthalpy_implicit,
    solid_preheating_profile,
    add_pre_melting_interval,
)


PLOTS_DIR = "plots"
RESULTS_DIR = "results"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


COLORS_METAL = {
    "Ag": "#9467bd",
    "Al": "#1f77b4",
    "Cu": "#d62728",
    "Ti": "#2ca02c",
}

LINESTYLES_METAL = {
    "Ag": ":",
    "Al": "--",
    "Cu": "-.",
    "Ti": "-",
}

COLORS_TI64 = {
    5: "#1f77b4",
    50: "#ff7f0e",
    500: "#2ca02c",
    5000: "#d62728",
}


# ============================================================
# Solver wrapper
# ============================================================

def call_solver(
    scheme,
    mat,
    I_laser,
    t_start,
    t_end,
    z_max,
    Nz,
    dT_mushy,
    safety,
    T_init,
    force_liquid_surface,
    save_times,
    verbose,
    implicit_dt_factor=5.0,
    nonlinear_iters=3,
):
    if scheme == "explicit":
        return solve_fdm_enthalpy_explicit(
            mat=mat,
            I_laser=I_laser,
            t_start=t_start,
            t_end=t_end,
            z_max=z_max,
            Nz=Nz,
            dT_mushy=dT_mushy,
            safety=safety,
            T_init=T_init,
            force_liquid_surface=force_liquid_surface,
            save_times=save_times,
            verbose=verbose,
        )

    if scheme == "implicit":
        return solve_fdm_enthalpy_implicit(
            mat=mat,
            I_laser=I_laser,
            t_start=t_start,
            t_end=t_end,
            z_max=z_max,
            Nz=Nz,
            dT_mushy=dT_mushy,
            safety=safety,
            dt_factor=implicit_dt_factor,
            T_init=T_init,
            force_liquid_surface=force_liquid_surface,
            save_times=save_times,
            nonlinear_iters=nonlinear_iters,
            verbose=verbose,
        )

    raise ValueError("scheme must be 'explicit' or 'implicit'")


# ============================================================
# 1. Ag / Al / Cu / Ti
# ============================================================

def run_long_materials(
    scheme="explicit",
    Nz=1500,
    dT_mushy=10.0,
    safety=0.35,
    implicit_dt_factor=5.0,
    nonlinear_iters=3,
    verbose=True,
):
    """
    FDM for Ag, Al, Cu, Ti at:
        I = 1e9 W/m²
        t = t_melt ... 10 s

    force_liquid_surface=True because this stage starts after
    the known melting time.
    """
    results = {}

    for name, mat in MATERIALS_LONG.items():
        print("\n" + "=" * 70)
        print(f"{scheme.upper()} FDM — {name} | I = {mat['I_laser']:.1e} W/m²")
        print("=" * 70)

        I_laser = mat["I_laser"]
        t_start = mat["t_melt"]
        t_end = mat["t_max"]

        z_max = 15.0 * np.sqrt(mat["alpha_s"] * t_end)
        z_grid = np.linspace(0.0, z_max, Nz)
        T_init = solid_preheating_profile(z_grid, mat, I_laser)

        start_wall = time.time()

        t_arr, S_arr, z, T_final, T_profiles = call_solver(
            scheme=scheme,
            mat=mat,
            I_laser=I_laser,
            t_start=t_start,
            t_end=t_end,
            z_max=z_max,
            Nz=Nz,
            dT_mushy=dT_mushy,
            safety=safety,
            T_init=T_init,
            force_liquid_surface=True,
            save_times=[t_end],
            verbose=verbose,
            implicit_dt_factor=implicit_dt_factor,
            nonlinear_iters=nonlinear_iters,
        )

        elapsed = time.time() - start_wall

        t_full, S_full = add_pre_melting_interval(
            t_arr,
            S_arr,
            mat["t_melt"],
            n_pre=200,
        )

        results[name] = {
            "t": t_full,
            "S": S_full,
            "z": z,
            "T_final": T_final,
            "T_init": T_init,
            "mat": mat,
            "scheme": scheme,
        }

        print(f"  elapsed = {elapsed:.2f} s")
        print(f"  final S = {S_full[-1] * 100:.6f} cm")

        np.savez(
            os.path.join(RESULTS_DIR, f"fdm_{scheme}_{name}.npz"),
            t=t_full,
            S=S_full,
            z=z,
            T_final=T_final,
            T_init=T_init,
        )

        plot_single_long_material(name, results[name])
        plot_temperature_profile_long(name, results[name])

    plot_combined_long_materials(results, scheme)
    print_summary_long(results, scheme)

    return results


def plot_single_long_material(name, res):
    t = res["t"]
    S = res["S"]
    mat = res["mat"]
    scheme = res["scheme"]

    plt.figure(figsize=(9, 5))

    plt.plot(
        t,
        S * 100.0,
        color=COLORS_METAL[name],
        linestyle=LINESTYLES_METAL[name],
        linewidth=3.0,
        label=f"{scheme.capitalize()} FDM — {name}",
    )

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Melt depth S(t) (cm)", fontsize=12)
    plt.title(f"{scheme.capitalize()} FDM — {name} | I = {mat['I_laser']:.1e} W/m²", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0.0, mat["t_max"])
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"fdm_{scheme}_{name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_temperature_profile_long(name, res):
    z = res["z"]
    T = res["T_final"]
    mat = res["mat"]
    scheme = res["scheme"]

    Tm = mat["Tm"]
    S_end = res["S"][-1]

    plt.figure(figsize=(9, 5))

    liquid_mask = z <= S_end
    solid_mask = z >= S_end

    if S_end > 0.0:
        plt.plot(z[liquid_mask] * 100, T[liquid_mask], "r-", linewidth=2.5, label="Liquid region")
        plt.plot(z[solid_mask] * 100, T[solid_mask], "b-", linewidth=2.5, label="Solid region")
        plt.axvline(S_end * 100, color="k", linestyle="--", linewidth=2.0, label=f"S = {S_end * 100:.2f} cm")
    else:
        plt.plot(z * 100, T, "b-", linewidth=2.5, label="Temperature")

    plt.axhline(Tm, color="gray", linestyle=":", linewidth=2.0, label=f"Tm = {Tm:.0f} K")

    plt.xlabel("z (cm)", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.title(f"{scheme.capitalize()} T(z) at t = {mat['t_max']} s — {name}", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"fdm_{scheme}_T_final_{name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_combined_long_materials(results, scheme):
    plt.figure(figsize=(10, 6))

    for name, res in results.items():
        plt.plot(
            res["t"],
            res["S"] * 100.0,
            color=COLORS_METAL[name],
            linestyle=LINESTYLES_METAL[name],
            linewidth=3.0,
            label=f"{scheme.capitalize()} FDM — {name}",
        )

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Melt depth S(t) (cm)", fontsize=12)
    plt.title(f"{scheme.capitalize()} FDM melt depth for Ag, Al, Cu, Ti | I = 1e9 W/m²", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0.0, 10.0)
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"fdm_{scheme}_Ag_Al_Cu_Ti_combined.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"\nSaved: {path}")


def print_summary_long(results, scheme):
    print("\n" + "=" * 70)
    print(f"SUMMARY — {scheme.upper()} FDM — Ag, Al, Cu, Ti")
    print("=" * 70)
    print(f"{'Material':<10} {'S_final (cm)':>15} {'t_melt table (s)':>20}")
    print("-" * 50)

    for name, res in results.items():
        mat = res["mat"]
        print(f"{name:<10} {res['S'][-1] * 100:>15.6f} {mat['t_melt']:>20.6e}")


# ============================================================
# 2. Ti-6Al-4V
# ============================================================

def run_ti64(
    scheme="explicit",
    Nz=700,
    dT_mushy=10.0,
    safety=0.35,
    implicit_dt_factor=5.0,
    nonlinear_iters=3,
    verbose=True,
):
    """
    FDM for Ti-6Al-4V at:
        I = 5, 50, 500, 5000 kW/cm²
        t = 0 ... 7 μs

    force_liquid_surface=False because simulation starts
    from room temperature at t=0.
    """
    results = {}

    for I_kW_cm2 in TI64_INTENSITIES_KW_CM2:
        print("\n" + "=" * 70)
        print(f"{scheme.upper()} FDM — Ti-6Al-4V | I = {I_kW_cm2} kW/cm²")
        print("=" * 70)

        I_eff = ti64_effective_intensity_W_m2(I_kW_cm2, I_scale=1000.0)

        t_start = 0.0
        t_end = TI64["t_max"]
        z_max = 15.0 * np.sqrt(TI64["alpha_s"] * t_end)

        start_wall = time.time()

        t_arr, S_arr, z, T_final, T_profiles = call_solver(
            scheme=scheme,
            mat=TI64,
            I_laser=I_eff,
            t_start=t_start,
            t_end=t_end,
            z_max=z_max,
            Nz=Nz,
            dT_mushy=dT_mushy,
            safety=safety,
            T_init=None,
            force_liquid_surface=False,
            save_times=[t_end],
            verbose=verbose,
            implicit_dt_factor=implicit_dt_factor,
            nonlinear_iters=nonlinear_iters,
        )

        elapsed = time.time() - start_wall

        results[I_kW_cm2] = {
            "t": t_arr,
            "S": S_arr,
            "z": z,
            "T_final": T_final,
            "I_eff": I_eff,
            "mat": TI64,
            "scheme": scheme,
        }

        print(f"  elapsed = {elapsed:.2f} s")
        print(f"  final S = {S_arr[-1] * 1e6:.6f} μm")

        np.savez(
            os.path.join(RESULTS_DIR, f"fdm_{scheme}_Ti64_{I_kW_cm2}kWcm2.npz"),
            t=t_arr,
            S=S_arr,
            z=z,
            T_final=T_final,
        )

        plot_single_ti64(I_kW_cm2, results[I_kW_cm2])

    plot_combined_ti64(results, scheme)
    print_summary_ti64(results, scheme)

    return results


def plot_single_ti64(I_kW_cm2, res):
    t = res["t"]
    S = res["S"]
    scheme = res["scheme"]

    plt.figure(figsize=(9, 5))

    plt.plot(
        t * 1e6,
        S * 1e6,
        color=COLORS_TI64[I_kW_cm2],
        linewidth=3.0,
        label=f"{scheme.capitalize()} FDM — {I_kW_cm2} kW/cm²",
    )

    plt.xlabel("Time (μs)", fontsize=12)
    plt.ylabel("Melt depth X(t) (μm)", fontsize=12)
    plt.title(f"{scheme.capitalize()} FDM — Ti-6Al-4V | I = {I_kW_cm2} kW/cm²", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0.0, TI64["t_max"] * 1e6)
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"fdm_{scheme}_Ti64_{I_kW_cm2}kWcm2.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_combined_ti64(results, scheme):
    plt.figure(figsize=(10, 6))

    for I_kW_cm2, res in results.items():
        plt.plot(
            res["t"] * 1e6,
            res["S"] * 1e6,
            color=COLORS_TI64[I_kW_cm2],
            linewidth=3.0,
            label=f"{I_kW_cm2} kW/cm²",
        )

    plt.xlabel("Time (μs)", fontsize=12)
    plt.ylabel("Melt depth X(t) (μm)", fontsize=12)
    plt.title(f"{scheme.capitalize()} FDM melt depth — Ti-6Al-4V", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0.0, TI64["t_max"] * 1e6)
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"fdm_{scheme}_Ti64_all_intensities.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"\nSaved: {path}")


def print_summary_ti64(results, scheme):
    print("\n" + "=" * 70)
    print(f"SUMMARY — {scheme.upper()} FDM — Ti-6Al-4V")
    print("=" * 70)
    print(f"{'Intensity':<20} {'S_final (μm)':>15}")
    print("-" * 40)

    for I_kW_cm2, res in results.items():
        print(f"{str(I_kW_cm2) + ' kW/cm²':<20} {res['S'][-1] * 1e6:>15.6f}")


# ============================================================
# 3. Compare explicit vs implicit
# ============================================================

def plot_explicit_vs_implicit_long(explicit_results, implicit_results):
    plt.figure(figsize=(10, 6))

    for name in explicit_results:
        plt.plot(
            explicit_results[name]["t"],
            explicit_results[name]["S"] * 100.0,
            color=COLORS_METAL[name],
            linestyle="-",
            linewidth=2.5,
            label=f"{name} explicit",
        )
        plt.plot(
            implicit_results[name]["t"],
            implicit_results[name]["S"] * 100.0,
            color=COLORS_METAL[name],
            linestyle="--",
            linewidth=2.5,
            label=f"{name} implicit",
        )

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Melt depth S(t) (cm)", fontsize=12)
    plt.title("Explicit vs Implicit FDM — Ag, Al, Cu, Ti", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.xlim(0.0, 10.0)
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "fdm_explicit_vs_implicit_long.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_explicit_vs_implicit_ti64(explicit_results, implicit_results):
    plt.figure(figsize=(10, 6))

    for I_kW_cm2 in explicit_results:
        plt.plot(
            explicit_results[I_kW_cm2]["t"] * 1e6,
            explicit_results[I_kW_cm2]["S"] * 1e6,
            color=COLORS_TI64[I_kW_cm2],
            linestyle="-",
            linewidth=2.5,
            label=f"{I_kW_cm2} explicit",
        )
        plt.plot(
            implicit_results[I_kW_cm2]["t"] * 1e6,
            implicit_results[I_kW_cm2]["S"] * 1e6,
            color=COLORS_TI64[I_kW_cm2],
            linestyle="--",
            linewidth=2.5,
            label=f"{I_kW_cm2} implicit",
        )

    plt.xlabel("Time (μs)", fontsize=12)
    plt.ylabel("Melt depth X(t) (μm)", fontsize=12)
    plt.title("Explicit vs Implicit FDM — Ti-6Al-4V", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.xlim(0.0, TI64["t_max"] * 1e6)
    plt.ylim(bottom=0.0)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "fdm_explicit_vs_implicit_Ti64.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def print_comparison_table_long(explicit_results, implicit_results):
    print("\n" + "=" * 75)
    print("EXPLICIT vs IMPLICIT — Ag, Al, Cu, Ti")
    print("=" * 75)
    print(f"{'Material':<10} {'Explicit (cm)':>15} {'Implicit (cm)':>15} {'Diff %':>12}")
    print("-" * 60)

    for name in explicit_results:
        e = explicit_results[name]["S"][-1] * 100.0
        im = implicit_results[name]["S"][-1] * 100.0
        diff = abs(e - im) / (abs(e) + 1e-30) * 100.0
        print(f"{name:<10} {e:>15.6f} {im:>15.6f} {diff:>12.3f}")


def print_comparison_table_ti64(explicit_results, implicit_results):
    print("\n" + "=" * 75)
    print("EXPLICIT vs IMPLICIT — Ti-6Al-4V")
    print("=" * 75)
    print(f"{'Intensity':<20} {'Explicit (μm)':>15} {'Implicit (μm)':>15} {'Diff %':>12}")
    print("-" * 65)

    for I_kW_cm2 in explicit_results:
        e = explicit_results[I_kW_cm2]["S"][-1] * 1e6
        im = implicit_results[I_kW_cm2]["S"][-1] * 1e6
        diff = abs(e - im) / (abs(e) + 1e-30) * 100.0
        print(f"{str(I_kW_cm2) + ' kW/cm²':<20} {e:>15.6f} {im:>15.6f} {diff:>12.3f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FDM baseline for 1D Stefan laser melting")
    print("Explicit and implicit enthalpy / effective heat capacity schemes")
    print("=" * 80)

    # For first verification keep implicit_dt_factor not too large.
    # Recommended checks:
    #   implicit_dt_factor = 1.0, 2.0, 5.0
    #   Nz = 700, 1500, 3000
    IMPLICIT_DT_FACTOR = 5.0
    NONLINEAR_ITERS = 3

    print("\n\n########## LONG MATERIALS: EXPLICIT ##########")
    long_explicit = run_long_materials(
        scheme="explicit",
        Nz=1500,
        dT_mushy=10.0,
        safety=0.35,
        verbose=True,
    )

    print("\n\n########## LONG MATERIALS: IMPLICIT ##########")
    long_implicit = run_long_materials(
        scheme="implicit",
        Nz=1500,
        dT_mushy=10.0,
        safety=0.35,
        implicit_dt_factor=IMPLICIT_DT_FACTOR,
        nonlinear_iters=NONLINEAR_ITERS,
        verbose=True,
    )

    print_comparison_table_long(long_explicit, long_implicit)
    plot_explicit_vs_implicit_long(long_explicit, long_implicit)

    print("\n\n########## Ti-6Al-4V: EXPLICIT ##########")
    ti64_explicit = run_ti64(
        scheme="explicit",
        Nz=700,
        dT_mushy=10.0,
        safety=0.35,
        verbose=True,
    )

    print("\n\n########## Ti-6Al-4V: IMPLICIT ##########")
    ti64_implicit = run_ti64(
        scheme="implicit",
        Nz=700,
        dT_mushy=10.0,
        safety=0.35,
        implicit_dt_factor=IMPLICIT_DT_FACTOR,
        nonlinear_iters=NONLINEAR_ITERS,
        verbose=True,
    )

    print_comparison_table_ti64(ti64_explicit, ti64_implicit)
    plot_explicit_vs_implicit_ti64(ti64_explicit, ti64_implicit)

    print("\nDone.")