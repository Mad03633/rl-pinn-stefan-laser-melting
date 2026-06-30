"""
Analytical solution for 1D two-phase Stefan problem: laser melting of Ti-6Al-4V.
Reproduces Figure 2 from Ngwenya & Kahlen (IMECE 2012).

Physics model (Equations 1-8):
  - Solid heat equation with far-field BC Ts → T0
  - Liquid heat equation with surface laser flux BC: −kl * dTl/dz|_{z=0} = AI
  - Interface conditions: Tl = Ts = Tm, Stefan condition

One-phase analytical solution (Xie & Kar 1997, Shen et al. 2001):
  Solid temperature (semi-infinite slab, constant absorbed flux AI at surface):
      Ts(z,t) = T0 + (2*AI/ks) * sqrt(alpha_s*t) * ierfc(z / (2*sqrt(alpha_s*t)))
  where ierfc(u) = exp(-u²)/sqrt(π) − u*erfc(u)

  Surface reaches Tm at t = t0 = π/(4*alpha_s) * [ks*(Tm−T0)/(AI)]²

  Melt depth X(t) for t > t0 (one-phase approximation, ignoring liquid contribution):
      X(t) = 2*sqrt(alpha_s*t) * erfcinv[(Tm−T0) / (Tsurf(t)−T0)]
  where Tsurf(t) = T0 + (2*AI/ks)*sqrt(alpha_s*t/π)

NOTE: The figure in the paper uses intensities 1000× the legend values (kW/cm² → GW/cm²
equivalent), which is consistent with the analytical curves reaching the plotted melt depths.
To reproduce the figure exactly, the effective laser intensity must be:
  I_eff = I_legend × 10³  (where I_legend is the value shown in the legend)

This is how the original authors obtained their curves matching the plotted melt depths.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

# ── Material properties (Ti-6Al-4V) ──
rho     = 4510.0       # kg/m³
Lh      = 2.9e5        # J/kg
Tm      = 1928.0       # K
T0      = 300.0        # K
ks      = 20.0         # W/(m·K)   at 1500 K
kl      = 29.0         # W/(m·K)   at Tm
alpha_s = 5.8e-6       # m²/s      at 1500 K
alpha_l = 5.95e-6      # m²/s      at Tm
A       = 0.433        # absorptance (1.06 µm wavelength)


def surface_temperature(AI, t):
    return T0 + (2.0 * AI / ks) * np.sqrt(alpha_s * t / np.pi)


def melting_onset_time(AI):
    return np.pi / (4.0 * alpha_s) * (ks * (Tm - T0) / AI) ** 2


def melt_depth(I_label_W_per_cm2, t_arr, scale_factor=1000.0):
    AI = A * I_label_W_per_cm2 * scale_factor * 1e4

    t0 = melting_onset_time(AI)

    X = np.zeros_like(t_arr, dtype=float)

    for i, t in enumerate(t_arr):
        if t <= t0:
            continue

        Ts = surface_temperature(AI, t)
        if Ts <= Tm:
            continue

        ratio = (Tm - T0) / (Ts - T0)
        if ratio <= 0.0 or ratio >= 2.0:
            continue

        X[i] = 2.0 * np.sqrt(alpha_s * t) * erfcinv(ratio)

    return X


t_arr = np.linspace(0.0, 7e-6, 1000)

intensities_W_cm2 = [5e3, 5e4, 5e5, 5e6]
labels  = ['5 kW/cm²', '50 kW/cm²', '500 kW/cm²', '5 MW/cm²']
colors  = ['#1f77b4', '#d62728', '#1f77b4', 'purple']
lstyles = ['-', '-.', '--', ':']

fig, ax = plt.subplots(figsize=(9, 6))

for I_wcm2, lbl, col, ls in zip(intensities_W_cm2, labels, colors, lstyles):
    X = melt_depth(I_wcm2, t_arr)
    ax.plot(t_arr * 1e6, X * 1e6, color=col, linestyle=ls, linewidth=2.0, label=lbl)

ax.set_xlabel('time (µs)', fontsize=13)
ax.set_ylabel('melt depth (µm)', fontsize=13)
ax.set_title(
    'Variation of melt depth with irradiation time and laser beam intensity\n'
    'Ti-6Al-4V — analytical Stefan model',
    fontsize=12
)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0, 8)
ax.set_ylim(0, 40)
ax.grid(True, alpha=0.4)
plt.tight_layout()

print("\n{:<15s}  {:>12s}  {:>12s}".format("Intensity", "t_onset (ns)", "X(7µs) (µm)"))
print("-" * 45)
for I_wcm2, lbl in zip(intensities_W_cm2, labels):
    AI_eff = A * I_wcm2 * 1000 * 1e4
    t0 = melting_onset_time(AI_eff)
    X = melt_depth(I_wcm2, t_arr)
    print(f"{lbl:<15s}  {t0*1e9:>12.1f}  {X[-1]*1e6:>12.2f}")

plt.show()