import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.special import erfcinv

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "lines.linewidth":   2.0,
})

METALS = {
    "Ag": dict(
        rho_s=10500, ks=429, kl=361,
        alpha_s=1.738602e-04, alpha_l=1.329356e-04,
        Tm=1234, T0=300.0, Lh=1.112e5,
        A_s=0.02, A_l=0.043, t_melt=1.15, t_max=10.0,
        I_laser=1e9,
        color="#9467bd", marker="o", label="Ag",
    ),
    "Al": dict(
        rho_s=2700, ks=238, kl=100,
        alpha_s=9.612666e-05, alpha_l=3.882289e-05,
        Tm=933, T0=300.0, Lh=3.880e5,
        A_s=0.0588, A_l=0.064, t_melt=0.034, t_max=10.0,
        I_laser=1e9,
        color="#1f77b4", marker="s", label="Al",
    ),
    "Cu": dict(
        rho_s=8960, ks=401, kl=342,
        alpha_s=1.159442e-04, alpha_l=8.906250e-05,
        Tm=1358, T0=300.0, Lh=2.047e5,
        A_s=0.02, A_l=0.058, t_melt=1.94, t_max=10.0,
        I_laser=1e9,
        color="#d62728", marker="^", label="Cu",
    ),
    "Ti": dict(
        rho_s=4500, ks=21.6, kl=20.28,
        alpha_s=9.090909e-06, alpha_l=7.049009e-06,
        Tm=1940, T0=300.0, Lh=3.650e5,
        A_s=0.257, A_l=0.433, t_melt=1.045e-3, t_max=10.0,
        I_laser=1e9,
        color="#2ca02c", marker="D", label="Ti",
    ),
}

TI64 = dict(
    rho=4510.0, ks=20.0, kl=29.0,
    alpha_s=5.8e-6, alpha_l=5.95e-6,
    Tm=1928.0, T0=300.0, Lh=2.9e5,
    A=0.433, I_scale=1000.0, t_max=7e-6,
)

TI64_INTENSITIES = {
    5:    {"color": "#1f77b4", "ls": "-",  "label": "5 kW/cm²"},
    50:   {"color": "#ff7f0e", "ls": "--", "label": "50 kW/cm²"},
    500:  {"color": "#2ca02c", "ls": "-.", "label": "500 kW/cm²"},
    5000: {"color": "#d62728", "ls": ":",  "label": "5000 kW/cm²"},
}

RESULTS_DIR = "results"
OUT_DIR     = os.path.join("plots", "comparison")


def analytic_metal(t_arr, mat):
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
        Tsurf = T0 + (2.0 * AI_l / ks) * np.sqrt(alpha_s * t / np.pi)
        if Tsurf <= Tm:
            continue
        ratio = (Tm - T0) / (Tsurf - T0)
        if 0.0 < ratio < 1.0:
            S[i] = 2.0 * np.sqrt(alpha_s * (t - t_melt)) * erfcinv(ratio)
    return S


def analytic_ngwenya(t_arr, AI_eff, ks, alpha_s, Tm, T0):
    t_arr  = np.asarray(t_arr, dtype=np.float64)
    X      = np.zeros_like(t_arr)
    t_melt = np.pi / (4.0 * alpha_s) * (ks * (Tm - T0) / AI_eff)**2
    for i, t in enumerate(t_arr):
        if t <= t_melt or t < 1e-30:
            continue
        Tsurf = T0 + (2.0 * AI_eff / ks) * np.sqrt(alpha_s * t / np.pi)
        if Tsurf <= Tm:
            continue
        ratio = (Tm - T0) / (Tsurf - T0)
        if 0.0 < ratio < 2.0:
            X[i] = 2.0 * np.sqrt(alpha_s * t) * erfcinv(ratio)
    return X


def load_fdm(name, explicit=True):
    scheme = "explicit" if explicit else "implicit"
    path   = os.path.join(RESULTS_DIR, f"fdm_{scheme}_{name}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {"t": d["t"], "S": d["S"]}


def load_fdm_ti64(I_kW, explicit=True):
    scheme = "explicit" if explicit else "implicit"
    path   = os.path.join(RESULTS_DIR, f"fdm_{scheme}_Ti64_{I_kW}kWcm2.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {"t": d["t"], "S": d["S"]}


def load_pinn(name, version="v2"):
    path = os.path.join(RESULTS_DIR, f"pinn_{version}_{name}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {"t": d["t"], "S": d["S"]}


def load_pinn_ti64(I_kW, version="v2"):
    path = os.path.join(RESULTS_DIR, f"pinn_{version}_Ti64_{I_kW}kWcm2.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {"t": d["t"], "S": d["S"]}


def metrics(S_pinn, t_pinn, S_ref, t_ref):
    S_interp  = np.interp(t_ref, t_pinn, S_pinn)
    err_final = abs(S_interp[-1] - S_ref[-1]) / (abs(S_ref[-1]) + 1e-30) * 100
    err_l2    = (np.linalg.norm(S_interp - S_ref) /
                 (np.linalg.norm(S_ref) + 1e-30)) * 100
    return err_final, err_l2


def plot_single_metal(name, ax_main, ax_err):
    mat   = METALS[name]
    color = mat["color"]
    t_pre = np.linspace(0.0, mat["t_melt"], 80)
    t_plot = np.linspace(mat["t_melt"], mat["t_max"], 800)

    S_anal = analytic_metal(t_plot, mat)
    t_full = np.concatenate([t_pre, t_plot])
    S_full = np.concatenate([np.zeros_like(t_pre), S_anal])
    ax_main.plot(t_full, S_full * 100,
                 color="black", lw=1.4, ls="--", alpha=0.5,
                 label="Analytical")

    # ── FDM explicit (reference) ─────────────────────────────
    fdm = load_fdm(name, explicit=True)
    if fdm is not None:
        ax_main.plot(fdm["t"], fdm["S"] * 100,
                     color=color, lw=2.2, ls="-",
                     label="FDM explicit (reference)")
    else:
        print(f"  [!] FDM data not found for {name}")

    fdm_imp = load_fdm(name, explicit=False)
    if fdm_imp is not None:
        ax_main.plot(fdm_imp["t"], fdm_imp["S"] * 100,
                     color=color, lw=1.4, ls=":", alpha=0.6,
                     label="FDM implicit")

    # ── PINN ──────────────────────────────────────────────
    pinn = load_pinn(name)
    if pinn is not None:
        t_p = np.concatenate([t_pre, pinn["t"]])
        S_p = np.concatenate([np.zeros_like(t_pre), pinn["S"]])
        ax_main.plot(t_p, S_p * 100,
                     color=color, lw=2.2, ls="-.",
                     label="PINN")

        # Error PINN−FDM
        if fdm is not None:
            mask  = fdm["t"] >= mat["t_melt"] - 1e-12
            t_ref = fdm["t"][mask]
            S_ref = fdm["S"][mask]
            S_interp = np.interp(t_ref, pinn["t"], pinn["S"])
            err_abs  = (S_interp - S_ref) * 100
            ax_err.plot(t_ref, err_abs,
                        color=color, lw=1.8)
            ax_err.fill_between(t_ref, err_abs, 0,
                                alpha=0.12, color=color)
            ef, el = metrics(pinn["S"], pinn["t"], S_ref, t_ref)
            ax_err.set_title(f"Δ(PINN−FDM)  |  Δfinal={ef:.1f}%  L2={el:.1f}%",
                             fontsize=9)
    else:
        ax_err.set_title("PINN: data not found", fontsize=9)
        ax_err.text(0.5, 0.5, "Run\nStefan_1D_2P_direct_metals_v2.py",
                    transform=ax_err.transAxes,
                    ha="center", va="center",
                    fontsize=9, color="gray")

    ax_main.axvline(mat["t_melt"], color="gray", lw=0.8, ls=":",
                    alpha=0.7, label=f"$t_{{melt}}={mat['t_melt']:.3g}$ с")
    ax_main.set_xlim(0, mat["t_max"])
    ax_main.set_ylim(bottom=0)
    ax_main.set_ylabel("S(t) (cm)")
    ax_main.set_title(f"{name}  |  I = {mat['I_laser']:.0e} W/m²", fontsize=11)
    ax_main.legend(fontsize=8, loc="upper left")

    ax_err.axhline(0, color="black", lw=0.7, ls="--")
    ax_err.set_xlim(0, mat["t_max"])
    ax_err.set_ylabel("Error (cm)")
    ax_err.set_xlabel("Time (s)")



def plot_metal_individual(name):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
        sharex=True,
    )
    plot_single_metal(name, ax1, ax2)
    ax1.set_xlabel("")

    legend_lines = [
        Line2D([0], [0], color="black",          lw=1.4, ls="--",  label="Analytical"),
        Line2D([0], [0], color=METALS[name]["color"], lw=2.2, ls="-",   label="FDM explicit (reference)"),
        Line2D([0], [0], color=METALS[name]["color"], lw=1.4, ls=":",   label="FDM implicit"),
        Line2D([0], [0], color=METALS[name]["color"], lw=2.2, ls="-.",  label="PINN"),
    ]
    ax1.legend(handles=legend_lines, fontsize=9, loc="upper left")

    fig.suptitle(
        f"Melt depth — {name}  |  I = {METALS[name]['I_laser']:.0e} W/m²\n"
        f"FDM explicit vs PINN vs Analytical",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"comparison_{name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metals_grid():
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    for i, name in enumerate(["Ag", "Al", "Cu", "Ti"]):
        row, col = divmod(i, 2)
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs[row, col],
            height_ratios=[3, 1],
            hspace=0.06,
        )
        ax_main = fig.add_subplot(gs_inner[0])
        ax_err  = fig.add_subplot(gs_inner[1], sharex=ax_main)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        plot_single_metal(name, ax_main, ax_err)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.4, ls="--", label="Analytical"),
        Line2D([0], [0], color="gray",  lw=2.2, ls="-",  label="FDM explicit (reference)"),
        Line2D([0], [0], color="gray",  lw=1.4, ls=":",  label="FDM implicit"),
        Line2D([0], [0], color="gray",  lw=2.2, ls="-.", label="PINN"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )
    fig.suptitle(
        "Ag / Al / Cu / Ti  |  I = 10⁹ W/m²\n"
        "FDM explicit (reference) · FDM implicit · PINN · Analytical",
        fontsize=12,
    )
    path = os.path.join(OUT_DIR, "comparison_all_metals_grid.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_ti64_grid():
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    for i, I_kW in enumerate([5, 50, 500, 5000]):
        row, col = divmod(i, 2)

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs[row, col],
            height_ratios=[3, 1],
            hspace=0.06,
        )

        ax1 = fig.add_subplot(gs_inner[0])
        ax2 = fig.add_subplot(gs_inner[1], sharex=ax1)
        plt.setp(ax1.get_xticklabels(), visible=False)

        mat = TI64
        cfg = TI64_INTENSITIES[I_kW]

        AI_eff = mat["A"] * float(I_kW) * 1e7 * mat["I_scale"]
        t_ref = np.linspace(0.0, mat["t_max"], 2000)
        X_ng = analytic_ngwenya(
            t_ref,
            AI_eff,
            mat["ks"],
            mat["alpha_s"],
            mat["Tm"],
            mat["T0"],
        )

        # Ngwenya analytical
        ax1.plot(
            t_ref * 1e6,
            X_ng * 1e6,
            color="black",
            lw=1.4,
            ls="--",
            alpha=0.5,
            label="Ngwenya",
        )

        # FDM explicit
        fdm = load_fdm_ti64(I_kW, explicit=True)
        if fdm is not None:
            ax1.plot(
                fdm["t"] * 1e6,
                fdm["S"] * 1e6,
                color=cfg["color"],
                lw=2.2,
                ls="-",
                label="FDM explicit",
            )

        # FDM implicit
        fdm_imp = load_fdm_ti64(I_kW, explicit=False)
        if fdm_imp is not None:
            ax1.plot(
                fdm_imp["t"] * 1e6,
                fdm_imp["S"] * 1e6,
                color=cfg["color"],
                lw=1.4,
                ls=":",
                alpha=0.6,
                label="FDM implicit",
            )

        # PINN
        pinn = load_pinn_ti64(I_kW)
        if pinn is not None:
            ax1.plot(
                pinn["t"] * 1e6,
                pinn["S"] * 1e6,
                color=cfg["color"],
                lw=2.2,
                ls="-.",
                label="PINN",
            )

            if fdm is not None:
                S_interp = np.interp(fdm["t"], pinn["t"], pinn["S"])
                err_abs = (S_interp - fdm["S"]) * 1e6

                ax2.plot(
                    fdm["t"] * 1e6,
                    err_abs,
                    color=cfg["color"],
                    lw=1.8,
                )
                ax2.fill_between(
                    fdm["t"] * 1e6,
                    err_abs,
                    0,
                    alpha=0.12,
                    color=cfg["color"],
                )

                ef, el = metrics(pinn["S"], pinn["t"], fdm["S"], fdm["t"])
                ax2.set_title(
                    f"Δ(PINN−FDM) | Δfinal={ef:.1f}%  L2={el:.1f}%",
                    fontsize=9,
                )

        ax2.axhline(0, color="black", lw=0.7, ls="--")

        ax1.set_xlim(0, mat["t_max"] * 1e6)
        ax1.set_ylim(bottom=0)
        ax1.set_ylabel("X(t) (μm)")
        ax1.set_title(f"Ti-6Al-4V | I = {I_kW} kW/cm²", fontsize=11)
        ax1.legend(fontsize=8, loc="upper left")

        ax2.set_xlabel("Time (μs)")
        ax2.set_ylabel("Error (μm)")

    fig.suptitle(
        "Ti-6Al-4V | FDM explicit · FDM implicit · PINN · Ngwenya",
        fontsize=12,
    )

    path = os.path.join(OUT_DIR, "comparison_Ti64_grid.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metals_combined():
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [METALS[n]["color"] for n in METALS]

    for name, color in zip(METALS.keys(), colors):
        mat   = METALS[name]
        t_pre = np.linspace(0.0, mat["t_melt"], 80)
        t_plot = np.linspace(mat["t_melt"], mat["t_max"], 800)

        # Analytical
        S_anal = analytic_metal(t_plot, mat)
        ax.plot(
            np.concatenate([t_pre, t_plot]),
            np.concatenate([np.zeros_like(t_pre), S_anal * 100]),
            color=color, lw=1.2, ls="--", alpha=0.45,
        )
        # FDM
        fdm = load_fdm(name, explicit=True)
        if fdm is not None:
            ax.plot(fdm["t"], fdm["S"] * 100,
                    color=color, lw=2.2, ls="-",
                    label=f"{name}")
        # PINN
        pinn = load_pinn(name)
        if pinn is not None:
            ax.plot(
                np.concatenate([t_pre, pinn["t"]]),
                np.concatenate([np.zeros_like(t_pre), pinn["S"] * 100]),
                color=color, lw=2.2, ls="-.",
            )

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.2, ls="--", alpha=0.5, label="Analytical (reference)"),
        Line2D([0], [0], color="black", lw=2.2, ls="-",             label="FDM explicit"),
        Line2D([0], [0], color="black", lw=2.2, ls="-.",            label="PINN"),
    ]
    mat_handles = [
        Line2D([0], [0], color=METALS[n]["color"], lw=3, label=n)
        for n in METALS
    ]
    ax.legend(
        handles=mat_handles + legend_handles,
        fontsize=9, ncol=2, loc="upper left",
    )
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Melt depth S(t) (cm)", fontsize=11)
    ax.set_title("Ag / Al / Cu / Ti  |  I = 10⁹ W/m²  |  FDM · PINN · Analytical", fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_all_metals_combined.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_ti64_individual(I_kW):
    mat = TI64
    cfg = TI64_INTENSITIES[I_kW]

    AI_eff = mat["A"] * float(I_kW) * 1e7 * mat["I_scale"]
    t_ref  = np.linspace(0.0, mat["t_max"], 2000)
    X_ng   = analytic_ngwenya(t_ref, AI_eff, mat["ks"], mat["alpha_s"],
                               mat["Tm"], mat["T0"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
        sharex=True,
    )

    # Ngwenya analytical
    ax1.plot(t_ref * 1e6, X_ng * 1e6,
             color="black", lw=1.4, ls="--", alpha=0.5,
             label="Ngwenya")

    # FDM explicit
    fdm = load_fdm_ti64(I_kW, explicit=True)
    if fdm is not None:
        ax1.plot(fdm["t"] * 1e6, fdm["S"] * 1e6,
                 color=cfg["color"], lw=2.2, ls="-",
                 label="FDM explicit (reference)")

    # FDM implicit
    fdm_imp = load_fdm_ti64(I_kW, explicit=False)
    if fdm_imp is not None:
        ax1.plot(fdm_imp["t"] * 1e6, fdm_imp["S"] * 1e6,
                 color=cfg["color"], lw=1.4, ls=":", alpha=0.6,
                 label="FDM implicit")

    # PINN
    pinn = load_pinn_ti64(I_kW)
    if pinn is not None:
        ax1.plot(pinn["t"] * 1e6, pinn["S"] * 1e6,
                 color=cfg["color"], lw=2.2, ls="-.",
                 label="PINN")
        if fdm is not None:
            S_interp = np.interp(fdm["t"], pinn["t"], pinn["S"])
            err_abs  = (S_interp - fdm["S"]) * 1e6
            ax2.plot(fdm["t"] * 1e6, err_abs,
                     color=cfg["color"], lw=1.8)
            ax2.fill_between(fdm["t"] * 1e6, err_abs, 0,
                             alpha=0.12, color=cfg["color"])
            ef, el = metrics(pinn["S"], pinn["t"], fdm["S"], fdm["t"])
            ax2.set_title(f"Δ(PINN−FDM)  |  Δfinal={ef:.1f}%  L2={el:.1f}%",
                          fontsize=9)
    else:
        ax2.text(0.5, 0.5, "PINN: run Stefan_1D_2P_direct_ti64_v2.py",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=9, color="gray")

    ax2.axhline(0, color="black", lw=0.7, ls="--")
    ax1.set_xlim(0, mat["t_max"] * 1e6)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("X(t) (μm)")
    ax1.set_title(f"Ti-6Al-4V  |  I = {I_kW} kW/cm²", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax2.set_xlabel("Time (μs)")
    ax2.set_ylabel("Error (μm)")

    fig.suptitle(
        f"Ti-6Al-4V  |  I = {I_kW} kW/cm²\n"
        "FDM explicit (reference) · FDM implicit · PINN · Ngwenya (reference)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"comparison_Ti64_{I_kW}kW.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_ti64_combined():
    mat = TI64
    fig, ax = plt.subplots(figsize=(10, 6))

    for I_kW, cfg in TI64_INTENSITIES.items():
        AI_eff = mat["A"] * float(I_kW) * 1e7 * mat["I_scale"]
        t_ref  = np.linspace(0.0, mat["t_max"], 2000)
        X_ng   = analytic_ngwenya(t_ref, AI_eff, mat["ks"], mat["alpha_s"],
                                   mat["Tm"], mat["T0"])

        # Analytical
        ax.plot(t_ref * 1e6, X_ng * 1e6,
                color=cfg["color"], lw=1.2, ls="--", alpha=0.45)

        # FDM
        fdm = load_fdm_ti64(I_kW, explicit=True)
        if fdm is not None:
            ax.plot(fdm["t"] * 1e6, fdm["S"] * 1e6,
                    color=cfg["color"], lw=2.2, ls="-",
                    label=cfg["label"])

        # PINN
        pinn = load_pinn_ti64(I_kW)
        if pinn is not None:
            ax.plot(pinn["t"] * 1e6, pinn["S"] * 1e6,
                    color=cfg["color"], lw=2.2, ls="-.")

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.2, ls="--", alpha=0.5,
               label="Ngwenya analytical (reference)"),
        Line2D([0], [0], color="black", lw=2.2, ls="-",
               label="FDM explicit"),
        Line2D([0], [0], color="black", lw=2.2, ls="-.",
               label="PINN"),
    ]
    int_handles = [
        Line2D([0], [0], color=TI64_INTENSITIES[I]["color"], lw=3,
               label=TI64_INTENSITIES[I]["label"])
        for I in TI64_INTENSITIES
    ]
    ax.legend(handles=int_handles + legend_handles, fontsize=9, ncol=2)
    ax.set_xlabel("Time (μs)", fontsize=11)
    ax.set_ylabel("Melt depth X(t) (μm)", fontsize=11)
    ax.set_title("Ti-6Al-4V | 5 / 50 / 500 / 5000 kW/cm²\n"
                 "FDM (reference) · PINN · Ngwenya analytical (reference)",
                 fontsize=11)
    ax.set_xlim(0, mat["t_max"] * 1e6)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_Ti64_all.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_metrics_table():
    print("\n" + "=" * 72)
    print("METRICS  |  PINN vs FDM  (main reference)")
    print("=" * 72)
    print(f"{'Material':<16} {'FDM S_final':>13} {'PINN S_final':>14} "
          f"{'Δfinal %':>10} {'L2 %':>8}")
    print("-" * 65)

    for name, mat in METALS.items():
        fdm  = load_fdm(name)
        pinn = load_pinn(name)
        if fdm is None:
            print(f"  {name}: FDM data not found")
            continue
        mask  = fdm["t"] >= mat["t_melt"] - 1e-12
        S_ref = fdm["S"][mask]
        t_ref = fdm["t"][mask]
        if pinn is None:
            print(f"  {name}: PINN data not found — run direct_metals_v2.py")
            continue
        ef, el = metrics(pinn["S"], pinn["t"], S_ref, t_ref)
        print(f"  {name:<14} {S_ref[-1]*100:>13.4f} cm  "
              f"{pinn['S'][-1]*100:>12.4f} cm  {ef:>8.2f}%  {el:>6.2f}%")

    print()
    print(f"{'Ti-6Al-4V':<16} {'FDM X_final':>13} {'PINN X_final':>14} "
          f"{'Δfinal %':>10} {'L2 %':>8}")
    print("-" * 65)
    for I_kW in TI64_INTENSITIES:
        fdm  = load_fdm_ti64(I_kW)
        pinn = load_pinn_ti64(I_kW)
        if fdm is None:
            print(f"  Ti64 {I_kW} kW/cm²: FDM data not found")
            continue
        if pinn is None:
            print(f"  Ti64 {I_kW} kW/cm²: PINN data not found")
            continue
        ef, el = metrics(pinn["S"], pinn["t"], fdm["S"], fdm["t"])
        print(f"  Ti64 {str(I_kW)+'kW/cm²':<11} {fdm['S'][-1]*1e6:>13.3f} μm  "
              f"{pinn['S'][-1]*1e6:>12.3f} μm  {ef:>8.2f}%  {el:>6.2f}%")
    print()


# ── Entry point ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Publication figures FDM / PINN / Analytical"
    )
    parser.add_argument("--metals-only", action="store_true")
    parser.add_argument("--ti64-only",   action="store_true")
    parser.add_argument("--material",    type=str, choices=["Ag", "Al", "Cu", "Ti"],
                        help="One metal")
    parser.add_argument("--intensity",   type=int, choices=[5, 50, 500, 5000],
                        help="One intensity for Ti-6Al-4V")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.material:
        print(f"\n── Individual plot: {args.material} ──")
        plot_metal_individual(args.material)

    elif args.intensity:
        print(f"\n── Ti-6Al-4V: {args.intensity} kW/cm² ──")
        plot_ti64_individual(args.intensity)

    elif args.metals_only:
        print("\n── Ag / Al / Cu / Ti ──")
        for name in METALS:
            print(f"  {name}...")
            plot_metal_individual(name)
        plot_metals_grid()
        plot_metals_combined()

    elif args.ti64_only:
        print("\n── Ti-6Al-4V ──")
        for I_kW in TI64_INTENSITIES:
            print(f"  {I_kW} kW/cm²...")
            plot_ti64_individual(I_kW)
        plot_ti64_grid()
        plot_ti64_combined()

    else:
        print("\n── Ag / Al / Cu / Ti ──")
        for name in METALS:
            print(f"  {name}...")
            plot_metal_individual(name)
        plot_metals_grid()
        plot_metals_combined()

        print("\n── Ti-6Al-4V ──")
        for I_kW in TI64_INTENSITIES:
            print(f"  {I_kW} kW/cm²...")
            plot_ti64_individual(I_kW)
        plot_ti64_grid()
        plot_ti64_combined()

    print_metrics_table()
    print(f"\nAll plots are in: {OUT_DIR}/")
    print("Done.")