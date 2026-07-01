# ============================================================
# Stefan_1D_2P_direct_ti64_v2.py
# PINN v2 для Ti-6Al-4V — все 4 интеnsивности
# Эталоны: FDM (главный) + Ngwenya (справка)
#
# ЗАПУСК:
#   python Stefan_1D_2P_direct_ti64_v2.py --intensity 5
#   python Stefan_1D_2P_direct_ti64_v2.py --intensity 50
#   python Stefan_1D_2P_direct_ti64_v2.py --intensity 500
#   python Stefan_1D_2P_direct_ti64_v2.py --intensity 5000
#   python Stefan_1D_2P_direct_ti64_v2.py --intensity all
# ============================================================

import argparse
import os
import time
from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt

from Stefan_1D_2P_models_ti64_v2 import (
    Stefan1D2P_v2,
    ngwenya_X, ngwenya_Ts, ngwenya_Tl,
    load_fdm_ti64, fdm_vs_ngwenya_report,
)

from save_pinn_results import save_pinn_ti64

# ── Параметры материала ───────────────────────────────────
MAT = dict(
    rho=4510.0, Lh=2.9e5, Tm=1928.0, T0=300.0,
    ks=20.0, kl=29.0,
    alpha_s=5.8e-6, alpha_l=5.95e-6,
    A=0.433, I_scale=1000.0,
    t_max=7e-6,
)

INTENSITIES = {
    5:    {"color": "#1f77b4", "ls": "-"},
    50:   {"color": "#ff7f0e", "ls": "--"},
    500:  {"color": "#2ca02c", "ls": "-."},
    5000: {"color": "#d62728", "ls": ":"},
}

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"


# ── Подготовка данных ─────────────────────────────────────
def make_data_ti64(z_max, t_max, X_ng, t_ref, fdm_ref,
                   AI_eff, ks, alpha_s, kl, Tm, T0,
                   Nr=25000, N0=8000, Nbc=8000, NX=8000,
                   N_sup_X=5000, N_sup_T=8000,
                   N_fdm_X=3000, N_fdm_T=3000,
                   seed=1234):
    rng   = np.random.RandomState(seed)
    t_eps = 1e-9

    def X_at(t_val):
        return float(np.interp(t_val, t_ref, X_ng))

    # Физика: liquid residual (z < X(t))
    t_rl = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    z_rl = np.array([rng.uniform(0.0, max(X_at(ti), 1e-9)) for ti in t_rl],
                    dtype=np.float32)

    # Физика: solid residual (z > X(t))
    t_rs = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    z_rs = np.array([rng.uniform(X_at(ti), z_max) for ti in t_rs],
                    dtype=np.float32)

    z0   = rng.uniform(0.0, z_max, (N0,  1)).astype(np.float32)
    t_bc = rng.uniform(t_eps, t_max, (Nbc, 1)).astype(np.float32)
    t_X  = rng.uniform(t_eps, t_max, (NX,  1)).astype(np.float32)

    # Ngwenya supervision X
    t_sup_X = rng.uniform(t_eps, t_max, N_sup_X).astype(np.float32)
    X_sup   = np.array([X_at(ti) for ti in t_sup_X], dtype=np.float32)

    # Ngwenya supervision Ts
    t_sup_Ts = rng.uniform(t_eps, t_max, N_sup_T).astype(np.float32)
    z_sup_Ts = np.array([rng.uniform(X_at(ti), z_max) for ti in t_sup_Ts],
                        dtype=np.float32)

    # Ngwenya supervision Tl
    t_sup_Tl = rng.uniform(t_eps, t_max, N_sup_T).astype(np.float32)
    z_sup_Tl = np.array([rng.uniform(0.0, max(X_at(ti), 1e-9)) for ti in t_sup_Tl],
                        dtype=np.float32)
    X_sup_Tl = np.array([X_at(ti) for ti in t_sup_Tl], dtype=np.float32)

    Ts_sup_vals = ngwenya_Ts(z_sup_Ts, t_sup_Ts, AI_eff, ks, alpha_s, Tm, T0).astype(np.float32)
    Tl_sup_vals = ngwenya_Tl(z_sup_Tl, X_sup_Tl, AI_eff, kl, Tm).astype(np.float32)

    # FDM supervision X (новое)
    t_fdm  = fdm_ref["t_fdm"]
    S_fdm  = fdm_ref["S_fdm"]
    idx_f  = rng.choice(len(t_fdm), size=min(N_fdm_X, len(t_fdm)), replace=False)
    t_fdm_X = t_fdm[idx_f].astype(np.float32)
    X_fdm   = S_fdm[idx_f].astype(np.float32)

    # FDM supervision Ts при t = t_max (новое)
    z_fdm    = fdm_ref["z_fdm"]
    T_fdm    = fdm_ref["T_fdm"]
    S_end    = S_fdm[-1]
    sol_mask = z_fdm >= S_end * 0.9
    z_sol    = z_fdm[sol_mask]
    T_sol    = T_fdm[sol_mask]
    if len(z_sol) > N_fdm_T:
        idx_t   = rng.choice(len(z_sol), size=N_fdm_T, replace=False)
        z_fdm_Ts = z_sol[idx_t].astype(np.float32)
        Ts_fdm   = T_sol[idx_t].astype(np.float32)
    else:
        z_fdm_Ts = z_sol.astype(np.float32)
        Ts_fdm   = T_sol.astype(np.float32)
    t_fdm_Ts = np.full(len(z_fdm_Ts), t_max, dtype=np.float32)

    return dict(
        z_rl=z_rl.reshape(-1, 1),   t_rl=t_rl.reshape(-1, 1),
        z_rs=z_rs.reshape(-1, 1),   t_rs=t_rs.reshape(-1, 1),
        z0=z0, t_bc=t_bc, t_X=t_X,
        t_sup_X=t_sup_X.reshape(-1, 1), X_sup=X_sup.reshape(-1, 1),
        z_sup_Ts=z_sup_Ts.reshape(-1, 1), t_sup_Ts=t_sup_Ts.reshape(-1, 1),
        Ts_sup=Ts_sup_vals.reshape(-1, 1),
        z_sup_Tl=z_sup_Tl.reshape(-1, 1), t_sup_Tl=t_sup_Tl.reshape(-1, 1),
        Tl_sup=Tl_sup_vals.reshape(-1, 1),
        t_fdm_X=t_fdm_X.reshape(-1, 1),   X_fdm=X_fdm.reshape(-1, 1),
        z_fdm_Ts=z_fdm_Ts.reshape(-1, 1), t_fdm_Ts=t_fdm_Ts.reshape(-1, 1),
        Ts_fdm=Ts_fdm.reshape(-1, 1),
    )


# ── Основная функция для одной интеnsивности ──────────────
def run_intensity(I_kW_cm2):
    mat   = MAT
    t_max = mat["t_max"]
    rho   = mat["rho"];     Lh      = mat["Lh"]
    Tm    = mat["Tm"];      T0      = mat["T0"]
    ks    = mat["ks"];      kl      = mat["kl"]
    alpha_s = mat["alpha_s"]; alpha_l = mat["alpha_l"]
    A     = mat["A"];       I_scale = mat["I_scale"]

    I_W_m2  = float(I_kW_cm2) * 1e7
    AI_eff  = A * I_W_m2 * I_scale
    z_max   = 15.0 * np.sqrt(alpha_s * t_max)

    print("\n" + "=" * 65)
    print(f"  PINN — Ti-6Al-4V  |  I = {I_kW_cm2} kW/cm²")
    print(f"  AI_eff = {AI_eff:.3e} W/m²")
    print("=" * 65)

    # FDM эталон
    fdm_path = os.path.join(RESULTS_DIR, f"fdm_explicit_Ti64_{I_kW_cm2}kWcm2.npz")
    if not os.path.exists(fdm_path):
        raise FileNotFoundError(
            f"FDM file not found: {fdm_path}\n"
            "Run first compare_fdm_pinn_analytical.py"
        )
    fdm_ref = load_fdm_ti64(fdm_path)
    fdm_X_max = float(fdm_ref['S_fdm'][-1])
    print(f"  FDM loaded: X_final = {fdm_ref['S_fdm'][-1]*1e6:.3f} μm")

    # Ngwenya эталон (справка)
    N_ref = 2000
    t_ref = np.linspace(0.0, t_max, N_ref)
    X_ng  = ngwenya_X(t_ref, AI_eff, ks, alpha_s, Tm, T0)
    X_max = float(X_ng.max())
    t_melt_ng = np.pi / (4.0 * alpha_s) * (ks * (Tm - T0) / AI_eff)**2

    fdm_vs_ngwenya_report(fdm_ref, X_ng, t_max)

    # Data
    data = make_data_ti64(
        z_max, t_max, X_ng, t_ref, fdm_ref,
        AI_eff, ks, alpha_s, kl, Tm, T0
    )
    print(f"\n  FDM sup: X_pts={len(data['t_fdm_X'])}, Ts_pts={len(data['z_fdm_Ts'])}")

    # Model
    model = Stefan1D2P_v2(
        z_min=0.0, z_max=z_max, t_min=0.0, t_max=t_max,
        rho=rho, Lh=Lh, T0=T0, Tm=Tm,
        ks=ks, kl=kl, alpha_s=alpha_s, alpha_l=alpha_l,
        A=A, I=I_W_m2, I_scale=I_scale,
        X_scale=None, X_max_hint=None, fdm_X_max=fdm_X_max,
        layers_T=(2, 128, 128, 128, 1),
        layers_X=(1, 128, 128, 128, 1),
        w_r=1.0, w_T0=10.0, w_bc=200.0, w_far=10.0,
        w_xt=800.0, w_xs=80.0, w_x0=10.0, w_xmin=30.0,
        X_min_m=5e-7,
        w_data_X=0.0, w_data_Ts=0.0, w_data_Tl=0.0,
        w_fdm_X=1000.0, w_fdm_Ts=80.0,
    )

    t0 = time.time()

    # Curriculum
    print("\n--- Phase 1: supervision warm-up (phys=0) ---")
    model.train(data, iters=8000, lr=5e-4, print_every=2000, phys_weight=0.0)

    print("\n--- Phase  2a: physics warm-in (phys=0.01) ---")
    model.train(data, iters=5000, lr=5e-4, print_every=1000, phys_weight=0.01)

    print("\n--- Phase 2b: полная физика + sup (phys=1) ---")
    model.train(data, iters=20000, lr=5e-4, print_every=2000, phys_weight=1.0)

    print("\n--- Phase 3: fine-tune lr=1e-4 ---")
    model.train(data, iters=15000, lr=1e-4, print_every=2000, phys_weight=1.0)

    print("\n--- Phase 4: polish lr=5e-5 ---")
    model.train(data, iters=8000, lr=5e-5, print_every=2000, phys_weight=0.1)

    print("\n--- Phase 5: restore physics lr=2e-5 ---")
    model.train(data, iters=5000, lr=2e-5, print_every=1000, phys_weight=1.0)

    elapsed = time.time() - t0
    print(f"\n  Learning time: {elapsed/60:.1f} min")

    # Metrics
    metrics = model.compute_fdm_metrics(fdm_ref)
    print(f"\n  ── Metrics (FDM) ──────────────────")
    print(f"  FDM  X(t_max) = {metrics['X_fdm_final']*1e6:.4f} μm")
    print(f"  PINN X(t_max) = {metrics['X_pinn_final']*1e6:.4f} μm")
    print(f"  Final error: {metrics['err_final_%']:.2f}%")
    print(f"  L2 error X(t): {metrics['err_l2_%']:.2f}%")
    print(f"\n  ── (Ngwenya) ────────")
    t_plot = np.linspace(0.0, t_max, 500).astype(np.float32).reshape(-1, 1)
    X_pinn = model.eval_X(t_plot)
    print(f"  Ngwenya X(t_max) = {X_max*1e6:.4f} μm")

    X_pinn_final = float(np.asarray(X_pinn[-1]).squeeze())
    X_ng_final = float(np.asarray(X_max).squeeze())

    # Save before any further printing/plotting, so the result is not lost.
    save_pinn_ti64(model, I_kW_cm2)

    print(f"  PINN vs Ngwenya:  {abs(X_pinn_final - X_ng_final) / X_ng_final * 100:.1f}%")

    # Plots
    _plot_ti64(I_kW_cm2, model, fdm_ref, X_ng, t_ref, t_plot, X_pinn, metrics, elapsed)

    return metrics, model


def _plot_ti64(I_kW_cm2, model, fdm_ref, X_ng, t_ref, t_plot, X_pinn, metrics, elapsed):
    col = INTENSITIES[I_kW_cm2]["color"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Ti-6Al-4V  I = {I_kW_cm2} kW/cm²  |  {elapsed/60:.1f} мин",
        fontsize=12
    )

    # Graph 1: X(t)
    ax = axes[0]
    ax.plot(fdm_ref["t_fdm"] * 1e6, fdm_ref["S_fdm"] * 1e6,
            'b-', lw=2.5, alpha=0.8, label='FDM')
    ax.plot(t_plot.flatten() * 1e6, X_pinn.flatten() * 1e6,
            color=col, ls='--', lw=2.5, label='PINN')
    ax.plot(t_ref * 1e6, X_ng * 1e6,
            'k--', lw=1.5, alpha=0.5, label='Ngwenya (reference)')
    ax.set_xlabel("Time (μs)"); ax.set_ylabel("X(t) (μm)")
    ax.set_title(f"X(t) | Δfinal={metrics['err_final_%']:.1f}% vs FDM")
    ax.legend(fontsize=9); ax.set_xlim(0, MAT["t_max"] * 1e6)
    ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3)

    # Graph 2: ошибка vs FDM
    ax2 = axes[1]
    t_fdm = fdm_ref["t_fdm"].astype(np.float32)
    S_fdm = fdm_ref["S_fdm"]
    X_pinn_on_fdm = model.eval_X(t_fdm).flatten()
    err = (X_pinn_on_fdm - S_fdm) * 1e6

    ax2.plot(fdm_ref["t_fdm"] * 1e6, err, color=col, lw=2.0)
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.fill_between(fdm_ref["t_fdm"] * 1e6, err, 0, alpha=0.15, color=col)
    ax2.set_xlabel("Time (μs)"); ax2.set_ylabel("X_pinn − X_fdm (μm)")
    ax2.set_title(f"Error vs FDM  |  L2={metrics['err_l2_%']:.1f}%")
    ax2.grid(True, alpha=0.3)

    # Graph 3: T(z) при t_max
    ax3 = axes[2]
    z_max  = fdm_ref["z_fdm"][-1]
    X_end  = X_pinn[-1, 0]
    z_liq  = np.linspace(0, X_end, 80).astype(np.float32)
    z_sol  = np.linspace(X_end, z_max * 0.5, 80).astype(np.float32)
    t_end  = np.full(80, MAT["t_max"], dtype=np.float32)

    ax3.plot(z_liq * 1e6, model.eval_Tl(z_liq, t_end).flatten(), 'r-', lw=2.0, label='Tl PINN')
    ax3.plot(z_sol * 1e6, model.eval_Ts(z_sol, t_end).flatten(), 'b-', lw=2.0, label='Ts PINN')
    ax3.plot(fdm_ref["z_fdm"] * 1e6, fdm_ref["T_fdm"], 'g--', lw=1.5, alpha=0.7, label='T FDM')
    ax3.axvline(X_end * 1e6, color='k', ls='--', lw=1.5, label=f'X={X_end*1e6:.2f} μm')
    ax3.axhline(MAT["Tm"], color='gray', ls=':', lw=1.5, label=f'Tm={MAT["Tm"]} K')
    ax3.set_xlabel("z (μm)"); ax3.set_ylabel("T (K)")
    ax3.set_title(f"T(z) при t = {MAT['t_max']*1e6:.0f} μs")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"pinn_v2_Ti64_{I_kW_cm2}kW_fdm_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Graph: {path}")


# ── Сводный график для всех интеnsивностей ────────────────
def plot_all_intensities(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))

    for I_kW_cm2, (metrics, model, fdm_ref, X_ng, t_ref) in all_results.items():
        col = INTENSITIES[I_kW_cm2]["color"]
        ls  = INTENSITIES[I_kW_cm2]["ls"]

        # FDM
        ax.plot(fdm_ref["t_fdm"] * 1e6, fdm_ref["S_fdm"] * 1e6,
                color=col, ls='-', lw=2.5, alpha=0.6,
                label=f'{I_kW_cm2} kW/cm² FDM')
        # PINN
        t_plot = np.linspace(0.0, MAT["t_max"], 400).astype(np.float32).reshape(-1, 1)
        X_pinn = model.eval_X(t_plot)
        ax.plot(t_plot.flatten() * 1e6, X_pinn.flatten() * 1e6,
                color=col, ls='--', lw=2.0,
                label=f'{I_kW_cm2} kW/cm² PINN')

    ax.set_xlabel("Time (μs)", fontsize=12)
    ax.set_ylabel("Depth of Melting X(t) (μm)", fontsize=12)
    ax.set_title("Ti-6Al-4V | PINN vs FDM | All Intensities", fontsize=12)
    ax.set_xlim(0, MAT["t_max"] * 1e6)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pinn_v2_Ti64_all_intensities.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"\nСводный график: {path}")


# ── Итоговая таблица ──────────────────────────────────────
def print_summary(all_metrics):
    print("\n" + "=" * 75)
    print("Results — Ti-6Al-4V | PINN vs FDM vs Ngwenya")
    print("=" * 75)
    print(f"{'I (kW/cm²)':<14} {'FDM X (μm)':>13} {'PINN X (μm)':>14} "
          f"{'Δfinal %':>10} {'L2 %':>8}")
    print("-" * 65)
    for I_kW, m in all_metrics.items():
        print(f"{str(I_kW):<14} {m['X_fdm_final']*1e6:>13.3f} {m['X_pinn_final']*1e6:>14.3f} "
              f"{m['err_final_%']:>10.2f} {m['err_l2_%']:>8.2f}")


# ── Точка входа ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensity", type=str, default="all",
                        choices=["5", "50", "500", "5000", "all"])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    intensities = [5, 50, 500, 5000] if args.intensity == "all" \
                  else [int(args.intensity)]

    all_results = {}
    all_metrics = {}

    for I_kW in intensities:
        metrics, model = run_intensity(I_kW)
        # Сохраняем для сводного графика
        fdm_ref = load_fdm_ti64(
            os.path.join(RESULTS_DIR, f"fdm_explicit_Ti64_{I_kW}kWcm2.npz")
        )
        AI = MAT["A"] * float(I_kW) * 1e7 * MAT["I_scale"]
        t_ref = np.linspace(0.0, MAT["t_max"], 2000)
        X_ng  = ngwenya_X(t_ref, AI, MAT["ks"], MAT["alpha_s"], MAT["Tm"], MAT["T0"])
        all_results[I_kW] = (metrics, model, fdm_ref, X_ng, t_ref)
        all_metrics[I_kW] = metrics

    print_summary(all_metrics)
    if len(intensities) > 1:
        plot_all_intensities(all_results)

    print("\nReady.")