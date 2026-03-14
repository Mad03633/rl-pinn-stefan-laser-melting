# Runner for PINN — 1-D two-phase Stefan, Ti-6Al-4V
# Ngwenya & Kahlen IMECE 2012

import numpy as np
import matplotlib.pyplot as plt
from Stefan_1D_2P_models import Stefan1D2P, ngwenya_X, ngwenya_Ts, ngwenya_Tl


def make_training_data(z_max, t_max, X_analytic_arr, t_ref_arr,
                       Nr=25000, N0=8000, Nbc=8000, NX=8000,
                       N_sup_X=5000, N_sup_T=8000,
                       seed=1234):
    rng = np.random.RandomState(seed)
    t_eps = 1e-9

    def X_at(t_val):
        return float(np.interp(t_val, t_ref_arr, X_analytic_arr))

    # liquid residual — z < X(t)
    t_rl = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    z_rl = np.array([rng.uniform(0.0, max(X_at(ti), 1e-9))
                     for ti in t_rl], dtype=np.float32)

    # solid residual — z > X(t)
    t_rs = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    z_rs = np.array([rng.uniform(X_at(ti), z_max)
                     for ti in t_rs], dtype=np.float32)

    z0   = rng.uniform(0.0, z_max, (N0,  1)).astype(np.float32)
    t_bc = rng.uniform(t_eps, t_max, (Nbc, 1)).astype(np.float32)
    t_X  = rng.uniform(t_eps, t_max, (NX,  1)).astype(np.float32)

    # supervision X
    t_sup_X = rng.uniform(t_eps, t_max, N_sup_X).astype(np.float32)
    X_sup   = np.array([X_at(ti) for ti in t_sup_X], dtype=np.float32)

    # supervision Ts (solid region)
    t_sup_Ts = rng.uniform(t_eps, t_max, N_sup_T).astype(np.float32)
    z_sup_Ts = np.array([rng.uniform(X_at(ti), z_max)
                         for ti in t_sup_Ts], dtype=np.float32)

    # supervision Tl (liquid region)
    t_sup_Tl = rng.uniform(t_eps, t_max, N_sup_T).astype(np.float32)
    z_sup_Tl = np.array([rng.uniform(0.0, max(X_at(ti), 1e-9))
                         for ti in t_sup_Tl], dtype=np.float32)
    X_sup_Tl = np.array([X_at(ti) for ti in t_sup_Tl], dtype=np.float32)

    return dict(
        z_rl=z_rl.reshape(-1,1), t_rl=t_rl.reshape(-1,1),
        z_rs=z_rs.reshape(-1,1), t_rs=t_rs.reshape(-1,1),
        z0=z0, t_bc=t_bc, t_X=t_X,
        t_sup_X=t_sup_X.reshape(-1,1), X_sup=X_sup.reshape(-1,1),
        z_sup_Ts=z_sup_Ts.reshape(-1,1), t_sup_Ts=t_sup_Ts.reshape(-1,1),
        Ts_sup=None,
        z_sup_Tl=z_sup_Tl.reshape(-1,1), t_sup_Tl=t_sup_Tl.reshape(-1,1),
        Tl_sup=None,
        _z_sup_Ts=z_sup_Ts, _t_sup_Ts=t_sup_Ts,
        _z_sup_Tl=z_sup_Tl, _t_sup_Tl=t_sup_Tl, _X_sup_Tl=X_sup_Tl,
    )


def blended_temperature(Tl, Ts, z_grid, t_grid, X_of_t):
    Nt  = X_of_t.shape[0]
    Nz  = z_grid.shape[0] // Nt
    return np.where(
        z_grid.reshape(Nz, Nt) < X_of_t.reshape(1, Nt),
        Tl.reshape(Nz, Nt), Ts.reshape(Nz, Nt)
    )


def main():
    # material properties Ti-6Al-4V
    rho = 4510.0
    Lh = 2.9e5
    Tm = 1928.0
    T0 = 300.0
    ks = 20.0
    kl = 29.0
    alpha_s = 5.8e-6
    alpha_l = 5.95e-6
    A = 0.433

    t_max = 7e-6
    z_max = 15.0 * np.sqrt(alpha_s * t_max)

    # Intensity
    I_label_W_cm2 = 5e3 # 5 kW/cm²
    I_W_per_m2 = I_label_W_cm2 * 1e4
    I_scale = 1000.0
    AI_eff = A * I_W_per_m2 * I_scale

    print("=" * 65)
    print("Ti-6Al-4V Laser Melting  —  PINN + Ngwenya supervision")
    print("=" * 65)
    print(f"  I = {I_label_W_cm2/1e3:.0f} kW/cm²   A·I_eff = {AI_eff:.3e} W/m²")

    N_ref = 2000
    t_ref = np.linspace(0.0, t_max, N_ref)
    X_ref = ngwenya_X(t_ref, AI_eff, ks, alpha_s, Tm, T0)
    X_max = float(X_ref.max())
    t_melt = np.pi / (4.0*alpha_s) * (ks*(Tm-T0)/AI_eff)**2
    print(f"  X_max = {X_max*1e6:.2f} µm   t_melt = {t_melt*1e9:.2f} ns")

    print("\nBuilding training data ...")
    data = make_training_data(z_max, t_max, X_ref, t_ref)

    Ts_sup_vals = ngwenya_Ts(
        data["_z_sup_Ts"], data["_t_sup_Ts"],
        AI_eff, ks, alpha_s, Tm, T0
    ).astype(np.float32)
    Tl_sup_vals = ngwenya_Tl(
        data["_z_sup_Tl"], data["_X_sup_Tl"],
        AI_eff, kl, Tm
    ).astype(np.float32)
    data["Ts_sup"] = Ts_sup_vals.reshape(-1,1)
    data["Tl_sup"] = Tl_sup_vals.reshape(-1,1)

    print(f"  Tl_sup range: [{Tl_sup_vals.min():.0f}, {Tl_sup_vals.max():.0f}] K")

    model = Stefan1D2P(
        z_min=0.0, z_max=z_max, t_min=0.0, t_max=t_max,
        rho=rho, Lh=Lh, T0=T0, Tm=Tm,
        ks=ks, kl=kl,
        alpha_s=alpha_s, alpha_l=alpha_l,
        A=A, I=I_W_per_m2,
        layers_T=(2, 128, 128, 128, 1),
        layers_X=(1, 128, 128, 128, 1),
        X_scale=z_max,
        I_scale=I_scale,
        X_max_hint=X_max,
        # physics weights
        w_r=1.0, w_T0=10.0, w_bc=200.0, w_far=10.0,
        w_xt=800.0, w_xs=80.0, w_x0=10.0, w_xmin=30.0,
        X_min_m=5e-7,
        # supervision weights
        w_data_X=500.0, w_data_Ts=50.0, w_data_Tl=50.0,
        seed=1234,
    )

    print("\n--- Phase 1: data supervision warm-up (phys_weight=0) ---")
    model.train(data, iters=8000, lr=5e-4, print_every=1000, phys_weight=0.0)

    print("\n--- Phase 2a: physics warm-in (phys_weight=0.01) ---")
    model.train(data, iters=5000, lr=5e-4, print_every=1000, phys_weight=0.01)

    print("\n--- Phase 2b: full physics + data (phys_weight=1) ---")
    model.train(data, iters=20000, lr=5e-4, print_every=1000, phys_weight=1.0)

    print("\n--- Phase 3: fine-tune lr=1e-4 ---")
    model.train(data, iters=15000, lr=1e-4, print_every=1000, phys_weight=1.0)

    print("\n--- Phase 4: polish (phys_weight=0.1, lr=5e-5) ---")
    model.train(data, iters=10000, lr=5e-5, print_every=1000, phys_weight=0.1)

    print("\n--- Phase 5: final physics restoration ---")
    model.train(data, iters=5000, lr=2e-5, print_every=1000, phys_weight=1.0)

    Nt = 500
    t_plot = np.linspace(0.0, t_max, Nt).astype(np.float32).reshape(-1,1)
    X_pinn = model.eval_X(t_plot)
    print(f"\nPINN   X(t_max) = {float(X_pinn[-1])*1e6:.4f} µm")
    print(f"Analyt X(t_max) = {X_max*1e6:.4f} µm")

    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(t_ref*1e6, X_ref*1e6,
             'k-', lw=2.5, label='Analytical (Ngwenya)')
    ax1.plot(t_plot.flatten()*1e6, X_pinn.flatten()*1e6,
             'r--', lw=2.5, label='PINN')
    ax1.set_xlabel("time (µs)", fontsize=13)
    ax1.set_ylabel("melt depth X(t) (µm)", fontsize=13)
    ax1.set_title(
        f"Melt depth vs time  (I = {I_label_W_cm2/1e3:.0f} kW/cm²)\n"
        "PINN vs Ngwenya & Kahlen analytical reference", fontsize=12)
    ax1.legend(fontsize=12);  ax1.grid(True, alpha=0.3)
    ymax = max(X_ref.max(), X_pinn.max()) * 1e6 * 1.15 + 0.5
    ax1.set_ylim(0, ymax)
    plt.tight_layout()

    Nz   = 160
    z_lin = np.linspace(0.0, z_max, Nz).astype(np.float32).reshape(-1,1)
    Zg   = np.repeat(z_lin, Nt, axis=0).astype(np.float32)
    Tg   = np.tile(t_plot, (Nz, 1)).astype(np.float32)
    T_bl = blended_temperature(
        model.eval_Tl(Zg, Tg), model.eval_Ts(Zg, Tg), Zg, Tg, X_pinn
    )
    z_show = max(X_max * 4, 5e-6)

    fig2, ax2 = plt.subplots(figsize=(9,5))
    cs = ax2.contourf(t_plot.flatten()*1e6, z_lin.flatten()*1e6,
                      T_bl, levels=60, cmap='inferno')
    plt.colorbar(cs, ax=ax2, label="Temperature (K)")
    ax2.plot(t_ref*1e6,             X_ref*1e6,
             'w-',  lw=2, label='Analytical X(t)')
    ax2.plot(t_plot.flatten()*1e6,  X_pinn.flatten()*1e6,
             'c--', lw=2, label='PINN X(t)')
    ax2.set_ylim(0, z_show*1e6)
    ax2.set_xlabel("time (µs)", fontsize=13)
    ax2.set_ylabel("z (µm)", fontsize=13)
    ax2.set_title("Temperature field — PINN prediction", fontsize=12)
    ax2.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(9,6))
    configs = [
        (5e3,  '5 kW/cm²',   '-',  '#1f77b4'),
        (5e4,  '50 kW/cm²',  '-.', '#d62728'),
        (5e5,  '500 kW/cm²', '--', '#2ca02c'),
        (5e6,  '5 MW/cm²',   ':',  '#9467bd'),
    ]
    for I_wcm2, lbl, ls, col in configs:
        AI_i = A * I_wcm2 * 1e4 * I_scale
        X_i  = ngwenya_X(t_ref, AI_i, ks, alpha_s, Tm, T0)
        ax3.plot(t_ref*1e6, X_i*1e6, color=col, linestyle=ls, lw=2, label=lbl)
    ax3.set_xlabel("time (µs)", fontsize=13)
    ax3.set_ylabel("melt depth (µm)", fontsize=13)
    ax3.set_title(
        "Melt depth variation  —  Ngwenya & Kahlen (IMECE 2012)\n"
        "Ti-6Al-4V, λ = 1.06 µm, A = 0.433", fontsize=12)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.set_xlim(0,8);  ax3.set_ylim(0,40)
    ax3.grid(True, alpha=0.4)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()