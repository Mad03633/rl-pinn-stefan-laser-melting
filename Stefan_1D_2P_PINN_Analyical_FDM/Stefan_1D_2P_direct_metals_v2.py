# ============================================================
# Stefan_1D_2P_direct_metals_v2.py
# PINN for Ag, Al, Cu, Ti — v2
# Эталон: FDM (results/fdm_explicit_*.npz)
# Analytical: только for спраinки
#
# ЗАПУСК:
#   python Stefan_1D_2P_direct_metals_v2.py --material Ag
#   python Stefan_1D_2P_direct_metals_v2.py --material Al
#   python Stefan_1D_2P_direct_metals_v2.py --material Cu
#   python Stefan_1D_2P_direct_metals_v2.py --material Ti
#   python Stefan_1D_2P_direct_metals_v2.py --material all
# ============================================================

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

from Stefan_1D_2P_models_metals_v2 import (
    StefanMetalsV2,
    preheating_Ts,
    load_fdm_reference,
    sample_fdm_supervision,
)
from save_pinn_results import save_pinn_metal

# ── Параметры материалоin ──────────────────────────────────
MATERIALS = {
    "Ag": dict(
        rho_s=10500, rho_l=9300, ks=429, kl=361,
        alpha_s=1.738602e-04, alpha_l=1.329356e-04,
        Tm=1234, T0=300.0, Lh=1.112e5,
        A_s=0.02, A_l=0.043, t_melt=1.15, t_max=10.0,
        I_laser=1e9,
        S_scale=0.0903, S_max_hint=0.0695,
        color="#9467bd", linestyle=":",
        # Веса — скорректироinаны for FDM supervision
        w_bc_l=500.0, w_xt=800.0, w_xs=150.0,
        w_sup_S=400.0, w_sup_Ts=50.0,
    ),
    "Al": dict(
        rho_s=2700, rho_l=2385, ks=238, kl=100,
        alpha_s=9.612666e-05, alpha_l=3.882289e-05,
        Tm=933, T0=300.0, Lh=3.880e5,
        A_s=0.0588, A_l=0.064, t_melt=0.034, t_max=10.0,
        I_laser=1e9,
        S_scale=0.1041, S_max_hint=0.0801,
        color="#1f77b4", linestyle="--",
        # Al: уinеличены inеса supervision, снижен w_bc_l
        w_bc_l=400.0, w_xt=800.0, w_xs=150.0,
        w_sup_S=600.0, w_sup_Ts=80.0,
    ),
    "Cu": dict(
        rho_s=8960, rho_l=8000, ks=401, kl=342,
        alpha_s=1.159442e-04, alpha_l=8.906250e-05,
        Tm=1358, T0=300.0, Lh=2.047e5,
        A_s=0.02, A_l=0.058, t_melt=1.94, t_max=10.0,
        I_laser=1e9,
        S_scale=0.0701, S_max_hint=0.054,
        color="#d62728", linestyle="-.",
        # Cu: усилен Stefan condition weight
        w_bc_l=500.0, w_xt=800.0, w_xs=300.0,
        w_sup_S=400.0, w_sup_Ts=50.0,
    ),
    "Ti": dict(
        rho_s=4500, rho_l=4110, ks=21.6, kl=20.28,
        alpha_s=9.090909e-06, alpha_l=7.049009e-06,
        Tm=1940, T0=300.0, Lh=3.650e5,
        A_s=0.257, A_l=0.433, t_melt=1.045e-3, t_max=10.0,
        I_laser=1e9,
        S_scale=0.0468, S_max_hint=0.036,
        color="#2ca02c", linestyle="-",
        # Ti: дополнительные BC-points через w_bc_l
        w_bc_l=600.0, w_xt=800.0, w_xs=150.0,
        w_sup_S=500.0, w_sup_Ts=60.0,
    ),
}

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"


# ── Аналитическое решение (кinазистационарное, for спраinки) ─
def analytic_S(t_arr, mat):
    """
    СПРАВОЧНОЕ решение — кinазистационарное atближение.
    Систематически заinышает S from-за:
      - использоinания alpha_s inместо alpha_l
      - неограниченного роста Tsurf
    Глаinный reference: FDM.
    """
    AI_l    = float(mat["A_l"]) * float(mat["I_laser"])
    ks      = float(mat["ks"])
    alpha_s = float(mat["alpha_s"])
    Tm      = float(mat["Tm"])
    T0      = float(mat["T0"])
    t_melt  = float(mat["t_melt"])

    S = np.zeros_like(t_arr, dtype=np.float64)
    for i, t in enumerate(t_arr):
        if t <= t_melt:
            continue
        tp     = t - t_melt
        Tsurf  = T0 + (2.0 * AI_l / ks) * np.sqrt(alpha_s * t / np.pi)
        if Tsurf <= Tm:
            continue
        ratio  = (Tm - T0) / (Tsurf - T0)
        if 0.0 < ratio < 1.0:
            S[i] = 2.0 * np.sqrt(alpha_s * tp) * erfcinv(ratio)
    return S


# ── Подготоinка обучающих данных ───────────────────────────
def make_data(mat, z_max, fdm_ref,
              Nr=15000, N0=3000, Nbc=2000, NX=2000,
              N_sup_S=3000, N_sup_T=3000, seed=42):
    """
    Формирует слоinарь обучающих данных.
    Ноinое: добаinляет FDM supervision points.
    """
    rng    = np.random.RandomState(seed)
    t_melt = float(mat["t_melt"])
    t_max  = float(mat["t_max"])
    t_eps  = t_melt + 1e-10
    S_scale = float(mat["S_scale"])
    alpha_s = float(mat["alpha_s"])
    ks      = float(mat["ks"])
    A_s     = float(mat["A_s"])
    I       = float(mat["I_laser"])
    Tm      = float(mat["Tm"])
    T0      = float(mat["T0"])

    # Residual points жидкой фазы — z in [0, S_scale*sqrt(tau)]
    t_rl   = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    tau_rl = (t_rl - t_melt) / (t_max - t_melt)
    z_rl_max = S_scale * np.sqrt(np.clip(tau_rl, 1e-9, None))
    z_rl   = (rng.uniform(0, 1, Nr) * z_rl_max).astype(np.float32)

    # Residual points тinёрдой фазы
    z_rs = rng.uniform(0.0, z_max, (Nr, 1)).astype(np.float32)
    t_rs = rng.uniform(t_eps, t_max, (Nr, 1)).astype(np.float32)

    # BC, интерфейс
    t_bc = rng.uniform(t_eps, t_max, (Nbc, 1)).astype(np.float32)
    t_S  = rng.uniform(t_eps, t_max, (NX,  1)).astype(np.float32)

    # IC: фfromически точный профиль прогреinа
    z_ic  = rng.uniform(0.0, z_max, (N0, 1)).astype(np.float32)
    Ts_ic = preheating_Ts(
        z_ic.flatten(), t_melt, A_s, I, ks, alpha_s, Tm, T0
    ).reshape(-1, 1)

    # FDM supervision
    fdm_sup = sample_fdm_supervision(
        fdm_ref, t_melt, t_max,
        N_sup_S=N_sup_S, N_sup_T=N_sup_T, seed=seed
    )

    return dict(
        z_rl=z_rl.reshape(-1, 1), t_rl=t_rl.reshape(-1, 1),
        z_rs=z_rs, t_rs=t_rs,
        z_ic=z_ic, Ts_ic=Ts_ic,
        t_bc=t_bc, t_S=t_S,
        **fdm_sup,
    )


# ── Вычисление PDE-нормалfromации from FDM ───────────────────
def compute_pde_scales_from_fdm(mat, fdm_ref):
    """
    Вычисляет корректные масштабы нормалfromации PDE-потерь
    from FDM данных, а не from эinристики.
    """
    t_fdm = fdm_ref["t_fdm"]
    S_fdm = fdm_ref["S_fdm"]
    T_fdm = fdm_ref["T_fdm"]

    Tm = float(mat["Tm"])
    T0 = float(mat["T0"])
    t_melt = float(mat["t_melt"])
    t_max  = float(mat["t_max"])
    alpha_l = float(mat["alpha_l"])
    alpha_s = float(mat["alpha_s"])

    # Теплоinой поток in жидкой фазе ~ AI_l / kl
    # Нормалfromуем inременным масштабом t_max - t_melt
    t_dur = t_max - t_melt

    # Масштаб тinёрдой фазы — прямо from данных
    pde_s_scale = (Tm - T0) / t_dur

    # Масштаб жидкой фазы — from максимальной T in жидкой зоне по FDM
    S_max = S_fdm[-1]
    # Жидкая зона: T > Tm in финальном профиле
    liquid_T = T_fdm[fdm_ref["z_fdm"] <= S_max * 1.05]
    if len(liquid_T) > 0:
        Tl_max = liquid_T.max()
        pde_l_scale = max((Tl_max - Tm) / t_dur, pde_s_scale)
    else:
        pde_l_scale = pde_s_scale

    return float(pde_l_scale), float(pde_s_scale)


# ── Осноinная функция for одного материала ─────────────────
def run_material(name, verbose=True):
    mat = MATERIALS[name]

    print("\n" + "=" * 65)
    print(f"  PINN — {name}  [FDM supervision]")
    print(f"  I = {mat['I_laser']:.0e} W/m²   t_melt = {mat['t_melt']:.4e} s")
    print("=" * 65)

    # Загрузка FDM referenceа
    fdm_path = os.path.join(RESULTS_DIR, f"fdm_explicit_{name}.npz")
    if not os.path.exists(fdm_path):
        raise FileNotFoundError(
            f"FDM file not found: {fdm_path}\n"
            f"Run compare_fdm_pinn_analytical.py first"
        )
    fdm_ref = load_fdm_reference(fdm_path)
    print(f"  FDM loaded: S_final = {fdm_ref['S_fdm'][-1]*100:.4f} cm")
    print(f"  FDM points: Nt={len(fdm_ref['t_fdm'])}, Nz={len(fdm_ref['z_fdm'])}")

    z_max = fdm_ref["z_fdm"][-1]

    # PDE нормалfromация from FDM
    pde_l, pde_s = compute_pde_scales_from_fdm(mat, fdm_ref)
    print(f"  PDE scales (from FDM): pde_l={pde_l:.2e}  pde_s={pde_s:.2e}")

    # Данные
    data = make_data(mat, z_max, fdm_ref)
    print(f"  IC Ts: [{data['Ts_ic'].min():.0f}, {data['Ts_ic'].max():.0f}] K")
    print(f"  FDM supervision: S_pts={len(data['t_sup_S'])}, Ts_pts={len(data['z_sup_Ts'])}")

    # Модель
    model = StefanMetalsV2(
        z_max=z_max, t_melt=mat["t_melt"], t_max=mat["t_max"],
        rho_s=mat["rho_s"], rho_l=mat["rho_l"],
        ks=mat["ks"], kl=mat["kl"],
        alpha_s=mat["alpha_s"], alpha_l=mat["alpha_l"],
        Lh=mat["Lh"], Tm=mat["Tm"], T0=mat["T0"],
        A_s=mat["A_s"], A_l=mat["A_l"], I=mat["I_laser"],
        S_scale=mat["S_scale"], S_max_hint=mat["S_max_hint"],
        pde_l_scale=pde_l, pde_s_scale=pde_s,
        layers_T=(2, 64, 64, 64, 1),
        layers_S=(1, 64, 64, 64, 1),
        w_r=1.0, w_ic=50.0,
        w_bc_l=mat["w_bc_l"], w_bc_s=20.0,
        w_xt=mat["w_xt"], w_xs=mat["w_xs"],
        w_x0=20.0, w_xmin=20.0,
        w_sup_S=mat["w_sup_S"], w_sup_Ts=mat["w_sup_Ts"],
        X_min_m=1e-8,
    )

    t0 = time.time()

    # ── Учебная программа (curriculum) ───────────────────
    # Phase 1: IC + FDM supervision, without physics
    print("\n--- Phase 1: IC + FDM supervision  lr=5e-4  phys=0.0  sup=1.0 ---")
    model.train(data, iters=5000, lr=5e-4, print_every=1000,
                phys_weight=0.0, sup_weight=1.0)

    # Phase 2a: add physics carefully
    print("\n--- Phase 2a: physics warm-in  lr=5e-4  phys=0.01  sup=1.0 ---")
    model.train(data, iters=5000, lr=5e-4, print_every=1000,
                phys_weight=0.01, sup_weight=1.0)

    # Phase 2b: полная physics + supervision
    print("\n--- Phase 2b: полная physics  lr=5e-4  phys=1.0  sup=1.0 ---")
    model.train(data, iters=20000, lr=5e-4, print_every=2000,
                phys_weight=1.0, sup_weight=1.0)

    # Phase 3: fine-tune
    print("\n--- Phase 3: fine-tune  lr=1e-4  phys=1.0  sup=0.5 ---")
    model.train(data, iters=15000, lr=1e-4, print_every=2000,
                phys_weight=1.0, sup_weight=0.5)

    # Phase 4: final polish (только physics)
    print("\n--- Phase 4: polish  lr=5e-5  phys=1.0  sup=0.2 ---")
    model.train(data, iters=10000, lr=5e-5, print_every=2000,
                phys_weight=1.0, sup_weight=0.2)

    elapsed = time.time() - t0
    print(f"\n  Learning time: {elapsed/60:.1f} min")

    # ── Metrics ───────────────────────────────────────────
    metrics = model.compute_fdm_metrics(fdm_ref, mat["t_melt"])
    print(f"\n  ── Metrics (reference: FDM) ──────────────────")
    print(f"  FDM  S(t_max) = {metrics['S_fdm_final']*100:.4f} cm")
    print(f"  PINN S(t_max) = {metrics['S_pinn_final']*100:.4f} cm")
    print(f"  Final Error: {metrics['err_final_%']:.2f}%")
    print(f"  L2 error S(t): {metrics['err_l2_%']:.2f}%")
    print(f"  Max error S: {metrics['err_max_m']*100:.4f} cm")

    # Analytical for спраinки
    t_melt_plot = np.linspace(mat["t_melt"], mat["t_max"], 500).astype(np.float32)
    S_pinn      = model.eval_S(t_melt_plot).flatten()
    S_anal      = analytic_S(t_melt_plot, mat)
    print(f"\n  ── Reference (аналит. quasi-steady) ────────")
    print(f"  Analytical S(t_max) = {S_anal[-1]*100:.4f} cm")
    print(f"  PINN vs analytical: {abs(S_pinn[-1]-S_anal[-1])/S_anal[-1]*100:.1f}%  "
          f"(Not main criterion)")

    # ── Графики ───────────────────────────────────────────
    t_pre = np.linspace(0.0, mat["t_melt"], 50)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{name}  |  I = {mat['I_laser']:.0e} W/m²  |  {elapsed/60:.1f} min",
        fontsize=12
    )

    # График 1: S(t) — сраinнение трёх методоin
    ax = axes[0]
    # FDM
    ax.plot(fdm_ref["t_fdm"], fdm_ref["S_fdm"] * 100,
            'b-', lw=2.5, alpha=0.8, label='FDM explicit (reference)')
    # PINN
    ax.plot(np.append(t_pre, t_melt_plot),
            np.append(np.zeros_like(t_pre), S_pinn * 100),
            color=mat["color"], ls=mat["linestyle"], lw=2.5,
            label=f'PINN — {name}')
    # Analytical (reference)
    ax.plot(np.append(t_pre, t_melt_plot),
            np.append(np.zeros_like(t_pre), S_anal * 100),
            'k--', lw=1.5, alpha=0.5, label='Analytical (reference)')
    ax.axvline(mat["t_melt"], color='gray', ls=':', lw=1.0)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Melt depth S(t) (cm)", fontsize=11)
    ax.set_title(f"S(t) — {name}\nFDM vs PINN vs Analytical", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, mat["t_max"])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # График 2: PINN vs FDM error
    ax2 = axes[1]
    t_fdm_valid = fdm_ref["t_fdm"][fdm_ref["t_fdm"] >= mat["t_melt"]]
    S_fdm_valid = fdm_ref["S_fdm"][fdm_ref["t_fdm"] >= mat["t_melt"]]
    S_pinn_interp = np.interp(t_fdm_valid,
                               t_melt_plot.astype(np.float64),
                               S_pinn.astype(np.float64))
    err_abs = (S_pinn_interp - S_fdm_valid) * 100  # in cm

    ax2.plot(t_fdm_valid, err_abs, color=mat["color"], lw=2.0)
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.fill_between(t_fdm_valid, err_abs, 0,
                     alpha=0.15, color=mat["color"])
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("S_pinn − S_fdm (cm)", fontsize=11)
    ax2.set_title(f"PINN vs FDM error\nL2={metrics['err_l2_%']:.1f}%", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # График 3: T(z) at t = t_max
    ax3 = axes[2]
    S_end = S_pinn[-1]
    z_liq = np.linspace(0, S_end, 80).astype(np.float32)
    z_sol = np.linspace(S_end, z_max * 0.25, 80).astype(np.float32)
    Tl_end = model.eval_Tl(z_liq, np.full(80, mat["t_max"], np.float32)).flatten()
    Ts_end = model.eval_Ts(z_sol, np.full(80, mat["t_max"], np.float32)).flatten()

    ax3.plot(z_liq * 100, Tl_end, 'r-', lw=2.0, label='Liquid Tl')
    ax3.plot(z_sol * 100, Ts_end, 'b-', lw=2.0, label='Solid Ts')
    # FDM температурный профиль
    ax3.plot(fdm_ref["z_fdm"] * 100, fdm_ref["T_fdm"],
             'g--', lw=1.5, alpha=0.7, label='FDM T(z)')
    ax3.axvline(S_end * 100, color='k', ls='--', lw=1.5,
                label=f'S = {S_end*100:.2f} cm')
    ax3.axhline(mat["Tm"], color='gray', ls=':', lw=1.5,
                label=f'Tm = {mat["Tm"]:.0f} K')
    ax3.set_xlabel("z (cm)", fontsize=11)
    ax3.set_ylabel("Temperature (K)", fontsize=11)
    ax3.set_title(f"T(z) at t = {mat['t_max']} с", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"pinn_v2_{name}_fdm_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"\n  Graph saved: {path}")

    save_pinn_metal(model, name, mat)

    return metrics, model


# ── Итогоinое сраinнение inсех материалоin ───────────────────
def print_final_summary(all_metrics):
    print("\n" + "=" * 80)
    print("FINAL COMPARISON — PINN vs FDM  (main reference)")
    print("=" * 80)
    print(f"{'Mat.':<6} {'FDM S (cm)':>12} {'PINN S (cm)':>12} "
          f"{'Δfinal %':>10} {'L2 %':>8} {'Max Δ (мм)':>12}")
    print("-" * 65)
    for name, m in all_metrics.items():
        print(f"{name:<6} {m['S_fdm_final']*100:>12.4f} {m['S_pinn_final']*100:>12.4f} "
              f"{m['err_final_%']:>10.2f} {m['err_l2_%']:>8.2f} "
              f"{m['err_max_m']*1000:>12.4f}")


# ── Сраinнительный график inсех материалоin ─────────────────
def plot_all_materials(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, (metrics, model)) in enumerate(all_results.items()):
        mat = MATERIALS[name]
        ax  = axes[i]

        fdm_ref = load_fdm_reference(
            os.path.join(RESULTS_DIR, f"fdm_explicit_{name}.npz")
        )
        t_pre        = np.linspace(0.0, mat["t_melt"], 50)
        t_melt_plot  = np.linspace(mat["t_melt"], mat["t_max"], 500).astype(np.float32)
        S_pinn       = model.eval_S(t_melt_plot).flatten()
        S_anal       = analytic_S(t_melt_plot, mat)

        ax.plot(fdm_ref["t_fdm"], fdm_ref["S_fdm"] * 100,
                'b-', lw=2.5, alpha=0.8, label='FDM (reference)')
        ax.plot(np.append(t_pre, t_melt_plot),
                np.append(np.zeros_like(t_pre), S_pinn * 100),
                color=mat["color"], ls=mat["linestyle"], lw=2.5,
                label=f'PINN — {name}')
        ax.plot(np.append(t_pre, t_melt_plot),
                np.append(np.zeros_like(t_pre), S_anal * 100),
                'k--', lw=1.2, alpha=0.4, label='Analytical (reference)')

        ax.set_title(f"{name} | Δfinal={metrics['err_final_%']:.1f}%"
                     f"  L2={metrics['err_l2_%']:.1f}%", fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("S(t) (cm)", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(0, mat["t_max"])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    plt.suptitle("PINN vs FDM vs Analytical (quasi-steady) | I = 1e9 W/m²",
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pinn_v2_all_metals_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"\nSummary plot: {path}")


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", type=str, default="all",
                        choices=["Ag", "Al", "Cu", "Ti", "all"],
                        help="Material for расчёта")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.material == "all":
        names = ["Ag", "Al", "Cu", "Ti"]
    else:
        names = [args.material]

    all_results = {}
    all_metrics = {}

    for name in names:
        metrics, model = run_material(name)
        all_results[name] = (metrics, model)
        all_metrics[name] = metrics

    print_final_summary(all_metrics)

    if len(names) > 1:
        plot_all_materials(all_results)

    print("\nDone.")