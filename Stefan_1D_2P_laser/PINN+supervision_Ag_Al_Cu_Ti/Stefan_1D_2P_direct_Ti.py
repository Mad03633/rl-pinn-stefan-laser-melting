# ============================================================
# Stefan_1D_2P_direct_Ti.py  — v4
# PINN + analytical supervision — Titanium
# I = 1e9 W/m²,  t in [t_melt=0.001045 s, 10 s]
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time
from Stefan_1D_2P_models_metals import (
    StefanMetals, k_S, k_Ts, k_Tl
)

# ── Material: Ti (Tables 1-3) ─────────────
rho_s   = 4500
rho_l   = 4110
ks      = 21.6
kl      = 20.28
alpha_s = 9.090909e-06
alpha_l = 7.054745e-06
Tm      = 1940
T0      = 300.0
Lh      = 3.650e+05
A_l     = 0.433
t_melt  = 1.045e-3

I_laser = 1e9
t_max   = 10.0
AI_l    = A_l * I_laser


def make_data(z_max, S_ref, t_ref,
              Nr=25000, N0=5000, Nbc=5000, NX=5000,
              N_sup=5000, N_sup_T=8000, seed=42):
    rng   = np.random.RandomState(seed)
    t_eps = t_melt + 1e-10

    def S_at(t_val):
        return float(np.interp(t_val, t_ref, S_ref))

    t_rl = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    z_rl = np.array([rng.uniform(0.0, max(S_at(ti), 1e-9))
                     for ti in t_rl], dtype=np.float32)
    t_rs = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    z_rs = np.array([rng.uniform(S_at(ti), z_max)
                     for ti in t_rs], dtype=np.float32)

    z_ic = rng.uniform(0.0, z_max,   (N0,  1)).astype(np.float32)
    t_bc = rng.uniform(t_eps, t_max,  (Nbc, 1)).astype(np.float32)
    t_X  = rng.uniform(t_eps, t_max,  (NX,  1)).astype(np.float32)

    t_sup_S  = rng.uniform(t_eps, t_max, N_sup).astype(np.float32)
    S_sup    = np.interp(t_sup_S, t_ref, S_ref).astype(np.float32)

    t_sup_Ts = rng.uniform(t_eps, t_max, N_sup_T).astype(np.float32)
    z_sup_Ts = np.array([rng.uniform(S_at(ti), z_max)
                         for ti in t_sup_Ts], dtype=np.float32)
    Ts_sup   = k_Ts(z_sup_Ts, t_sup_Ts, AI_l, ks, alpha_s, Tm, T0, t_melt)

    t_sup_Tl = rng.uniform(t_eps, t_max, N_sup_T).astype(np.float32)
    z_sup_Tl = np.array([rng.uniform(0.0, max(S_at(ti), 1e-9))
                         for ti in t_sup_Tl], dtype=np.float32)
    S_sup_Tl = np.interp(t_sup_Tl, t_ref, S_ref).astype(np.float32)
    Tl_sup   = k_Tl(z_sup_Tl, S_sup_Tl, AI_l, kl, Tm)

    return dict(
        z_rl=z_rl.reshape(-1,1),  t_rl=t_rl.reshape(-1,1),
        z_rs=z_rs.reshape(-1,1),  t_rs=t_rs.reshape(-1,1),
        z_ic=z_ic, t_bc=t_bc, t_X=t_X,
        t_sup_S=t_sup_S.reshape(-1,1),    S_sup=S_sup.reshape(-1,1),
        z_sup_Ts=z_sup_Ts.reshape(-1,1),  t_sup_Ts=t_sup_Ts.reshape(-1,1),
        Ts_sup=Ts_sup.reshape(-1,1),
        z_sup_Tl=z_sup_Tl.reshape(-1,1),  t_sup_Tl=t_sup_Tl.reshape(-1,1),
        Tl_sup=Tl_sup.reshape(-1,1),
    )


def main():
    print("=" * 60)
    print("  PINN + supervision — Ti")
    print(f"  I = {I_laser:.0e} W/m²   t_melt = {t_melt:.4e} s")
    print("=" * 60)

    z_max = 15.0 * np.sqrt(alpha_s * t_max)
    print(f"  z_max = {z_max*100:.2f} cm")

    t_ref = np.linspace(t_melt, t_max, 2000)
    S_ref = k_S(t_ref, AI_l, ks, alpha_s, Tm, T0, t_melt)
    S_max = float(S_ref.max())
    print(f"  S_max (analytical) = {S_max*100:.3f} cm")

    print("\nBuilding training data ...")
    data = make_data(z_max, S_ref, t_ref)
    print(f"  Tl_sup range: [{data['Tl_sup'].min():.0f}, {data['Tl_sup'].max():.0f}] K")

    model = StefanMetals(
        z_max=z_max, t_melt=t_melt, t_max=t_max,
        rho_s=rho_s, rho_l=rho_l, ks=ks, kl=kl,
        alpha_s=alpha_s, alpha_l=alpha_l,
        Lh=Lh, Tm=Tm, T0=T0, A_l=A_l, I=I_laser,
        S_max_hint=S_max,
        layers_T=(2, 128, 128, 128, 1),
        layers_S=(1, 128, 128, 128, 1),
        w_r=1.0, w_ic=10.0, w_bc_l=200.0, w_bc_s=10.0,
        w_xt=800.0, w_xs=80.0, w_x0=10.0, w_xmin=30.0,
        X_min_m=1e-7,
        w_data_S=500.0, w_data_Ts=50.0, w_data_Tl=50.0,
    )

    t0_train = time.time()

    print("\n--- Phase 1: supervision only   lr=5e-4  phys=0.0 ---")
    model.train(data, iters=8000,  lr=5e-4, print_every=1000, phys_weight=0.0)

    print("\n--- Phase 2a: physics warm-in   lr=5e-4  phys=0.01 ---")
    model.train(data, iters=5000,  lr=5e-4, print_every=1000, phys_weight=0.01)

    print("\n--- Phase 2b: full physics      lr=5e-4  phys=1.0 ---")
    model.train(data, iters=20000, lr=5e-4, print_every=2000, phys_weight=1.0)

    print("\n--- Phase 3: fine-tune          lr=1e-4  phys=1.0 ---")
    model.train(data, iters=15000, lr=1e-4, print_every=2000, phys_weight=1.0)

    print("\n--- Phase 4: data polish        lr=5e-5  phys=0.1 ---")
    model.train(data, iters=8000,  lr=5e-5, print_every=2000, phys_weight=0.1)

    print("\n--- Phase 5: physics restore    lr=2e-5  phys=1.0 ---")
    model.train(data, iters=5000,  lr=2e-5, print_every=1000, phys_weight=1.0)

    elapsed = time.time() - t0_train
    print(f"\n  Training time: {elapsed/60:.1f} min")

    t_plot = np.linspace(t_melt, t_max, 500).astype(np.float32).reshape(-1,1)
    S_pinn = model.eval_S(t_plot).flatten()
    S_anal = k_S(t_plot.flatten(), AI_l, ks, alpha_s, Tm, T0, t_melt)
    print(f"  PINN   S(t_max) = {S_pinn[-1]*100:.4f} cm")
    print(f"  Analyt S(t_max) = {S_anal[-1]*100:.4f} cm")
    print(f"  Error  at t_max = {abs(S_pinn[-1]-S_anal[-1])/(S_anal[-1]+1e-12)*100:.2f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_plot.flatten(), S_anal, 'k-',  lw=2.5, label='Analytical')
    ax.plot(t_plot.flatten(), S_pinn, color='#2ca02c', ls='-', lw=2.5, label='PINN — Ti')
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Melt depth S(t)  (m)", fontsize=13)
    ax.set_title("Melt depth — Ti\nI = 1e+09 W/m²,  PINN vs Analytical", fontsize=12)
    ax.legend(fontsize=12); ax.set_xlim(0, t_max); ax.set_ylim(0); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("melt_depth_Ti_PINN.png", dpi=150)
    print("  Figure saved: melt_depth_Ti_PINN.png")
    plt.show()


if __name__ == "__main__":
    main()