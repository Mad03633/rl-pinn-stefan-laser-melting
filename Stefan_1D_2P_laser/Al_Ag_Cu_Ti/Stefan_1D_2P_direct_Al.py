# ============================================================
# Stefan_1D_2P_direct_Al.py
# PINN pure physics — Al
# I = 1e9 W/m²,  t in [t_melt=0.034, 10 s]
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import erfcinv
from Stefan_1D_2P_models_metals import StefanMetals, preheating_Ts

# ── Params: Al
rho_s = 2700
rho_l = 2385
ks = 238
kl = 100
alpha_s = 9.612666e-05
alpha_l = 3.882289e-05
Tm = 933
T0 = 300.0
Lh = 3.880e+05
A_s = 0.0588
A_l = 0.064
t_melt = 0.034

I_laser = 1e9
t_max = 10.0
AI_l = A_l * I_laser


def analytic_S(t_arr):
    S = np.zeros_like(t_arr, dtype=np.float64)
    for i, t in enumerate(t_arr):
        tp = t - t_melt
        if tp <= 0: continue
        Tsurf = T0 + (2*AI_l/ks)*np.sqrt(alpha_s*tp/np.pi)
        if Tsurf <= Tm: continue
        ratio = (Tm-T0)/(Tsurf-T0)
        if 0 < ratio < 1:
            S[i] = 2*np.sqrt(alpha_s*tp)*erfcinv(ratio)
    return S


def make_data(z_max, Nr=15000, N0=3000, Nbc=2000, NX=2000, seed=42):
    rng = np.random.RandomState(seed)
    t_eps = t_melt + 1e-10

    t_rl = rng.uniform(t_eps, t_max, Nr).astype(np.float32)
    # z_rl in [0, S_scale*sqrt(tau)] — covers the entire liquid zone
    # The old sqrt(alpha_l*(t-t_melt)) method underestimated by 4x -> the network didn't see z>2cm
    tau_rl = (t_rl - t_melt) / (t_max - t_melt)
    z_rl_max = 0.1041 * np.sqrt(tau_rl).clip(1e-9)
    z_rl = (rng.uniform(0, 1, Nr) * z_rl_max).astype(np.float32)

    z_rs = rng.uniform(0.0, z_max,   (Nr,  1)).astype(np.float32)
    t_rs = rng.uniform(t_eps, t_max,  (Nr,  1)).astype(np.float32)
    t_bc = rng.uniform(t_eps, t_max,  (Nbc, 1)).astype(np.float32)
    t_S  = rng.uniform(t_eps, t_max,  (NX,  1)).astype(np.float32)

    z_ic = rng.uniform(0.0, z_max, (N0, 1)).astype(np.float32)
    Ts_ic = preheating_Ts(
        z_ic.flatten(), t_melt, A_s, I_laser, ks, alpha_s, Tm, T0
    ).reshape(-1, 1)

    return dict(
        z_rl=z_rl.reshape(-1,1), t_rl=t_rl.reshape(-1,1),
        z_rs=z_rs, t_rs=t_rs,
        z_ic=z_ic, Ts_ic=Ts_ic,
        t_bc=t_bc, t_S=t_S,
    )


def main():
    print("=" * 60)
    print("  PINN pure physics — Al  [full training]")
    print(f"  I = {I_laser:.0e} W/m²   t_melt = {t_melt:.4e} s")
    print("=" * 60)

    z_max = 15.0 * np.sqrt(alpha_s * t_max)
    print(f"  z_max = {z_max*100:.2f} cm")

    data = make_data(z_max)
    print(f"  IC Ts: [{data['Ts_ic'].min():.0f}, {data['Ts_ic'].max():.0f}] K")
    print(f"  z_rl max: {data['z_rl'].max()*100:.3f} cm")

    model = StefanMetals(
        z_max=z_max, t_melt=t_melt, t_max=t_max,
        rho_s=rho_s, rho_l=rho_l, ks=ks, kl=kl,
        alpha_s=alpha_s, alpha_l=alpha_l,
        Lh=Lh, Tm=Tm, T0=T0,
        A_s=A_s, A_l=A_l, I=I_laser,
        S_scale=0.1041,
        S_max_hint=0.0801,
        layers_T=(2, 64, 64, 64, 1),
        layers_S=(1, 64, 64, 64, 1),
        w_r=1.0, w_ic=50.0, w_bc_l=1000.0, w_bc_s=20.0,
        w_xt=1000.0, w_xs=500.0, w_x0=20.0, w_xmin=20.0,
        X_min_m=1e-8,
    )

    t0 = time.time()

    print("\n--- Phase 1: IC only      lr=5e-4  phys=0.0 ---")
    model.train(data, iters=5000,  lr=5e-4, print_every=1000, phys_weight=0.0)

    print("\n--- Phase 2a: warm-in     lr=5e-4  phys=0.01 ---")
    model.train(data, iters=5000,  lr=5e-4, print_every=1000, phys_weight=0.01)

    print("\n--- Phase 2b: medium phys lr=5e-4  phys=0.1 ---")
    model.train(data, iters=5000,  lr=5e-4, print_every=1000, phys_weight=0.1)

    print("\n--- Phase 2c: full phys   lr=5e-4  phys=1.0 ---")
    model.train(data, iters=20000, lr=5e-4, print_every=2000, phys_weight=1.0)

    print("\n--- Phase 3: fine-tune    lr=1e-4  phys=1.0 ---")
    model.train(data, iters=15000, lr=1e-4, print_every=2000, phys_weight=1.0)

    print("\n--- Phase 4: polish       lr=5e-5  phys=1.0 ---")
    model.train(data, iters=10000, lr=5e-5, print_every=2000, phys_weight=1.0)

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed/60:.1f} min")

    # ── Results 
    # Full range [0, t_max]: S=0 before t_melt, melting after
    t_pre = np.linspace(0.0, t_melt, 50)
    t_melt_plot = np.linspace(t_melt, t_max, 500).astype(np.float32)

    S_pinn = model.eval_S(t_melt_plot).flatten()
    S_anal = analytic_S(t_melt_plot)

    print(f"  PINN   S(t_max) = {S_pinn[-1]*100:.4f} cm")
    print(f"  Analyt S(t_max) = {S_anal[-1]*100:.4f} cm")
    print(f"  Deviation: {abs(S_pinn[-1]-S_anal[-1])/S_anal[-1]*100:.1f}%")

    # ── Graph
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    # S=0 until melting begins
    # Analytics: S=0 until t_melt, then growth
    ax.plot(np.append(t_pre, t_melt_plot),
            np.append(np.zeros_like(t_pre), S_anal*100),
            'k-', lw=2.0, alpha=0.6, label='Quasi-steady (analytical)')
    # PINN: explicitly add a point (t_melt, 0) to the beginning
    ax.plot(np.append(t_pre, t_melt_plot),
            np.append(np.zeros_like(t_pre), S_pinn*100),
            color='#1f77b4', ls='--', lw=2.5, label='PINN — Al')
    ax.axvline(t_melt, color='gray', ls='--', lw=1.0, alpha=0.6,
               label=f't_melt = {t_melt:.3g} s')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Melt depth S(t)  (cm)", fontsize=12)
    ax.set_title("Melt depth — Al\nPINN vs Analytical", fontsize=11)
    ax.legend(fontsize=11); ax.set_xlim(0, t_max); ax.set_ylim(0); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    S_end = S_pinn[-1]
    z_liq = np.linspace(0, S_end,        100).astype(np.float32)
    z_sol = np.linspace(S_end, z_max*0.3, 100).astype(np.float32)
    Tl_end = model.eval_Tl(z_liq, np.full(100, t_max, np.float32)).flatten()
    Ts_end = model.eval_Ts(z_sol, np.full(100, t_max, np.float32)).flatten()
    ax2.plot(z_liq*100, Tl_end, 'r-', lw=2.0, label='Liquid Tl')
    ax2.plot(z_sol*100, Ts_end, 'b-', lw=2.0, label='Solid Ts')
    ax2.axvline(S_end*100, color='k', ls='--', lw=1.5, label=f'S = {S_end*100:.2f} cm')
    ax2.axhline(Tm, color='gray', ls=':', lw=1.0, label=f'Tm = 933 K')
    ax2.set_xlabel("z (cm)", fontsize=12)
    ax2.set_ylabel("Temperature (K)", fontsize=12)
    ax2.set_title(f"T(z) at t = {t_max} s", fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Al  |  I = {I_laser:.0e} W/m²  |  {elapsed/60:.1f} min",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("melt_depth_Al_PINN_full.png", dpi=150, bbox_inches='tight')
    print("  Saved: melt_depth_Al_PINN_full.png")
    plt.show()


if __name__ == "__main__":
    main()