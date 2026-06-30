# fdm_solver.py
# ============================================================
# Explicit and implicit FDM solvers for 1D Stefan laser melting
# using fixed-grid effective heat capacity / enthalpy formulation.
#
# Explicit scheme:
#   T_i^{n+1} = T_i^n + dt * alpha_i^n *
#               (T_{i+1}^n - 2T_i^n + T_{i-1}^n) / dz^2
#
# Implicit scheme:
#   T_i^{n+1} = T_i^n + dt * alpha_i^n *
#               (T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}) / dz^2
#
# Phase change:
#   c_eff = c + L / dT_mushy  near Tm
#
# Surface heat flux:
#   -k dT/dz = A I
#
# Far boundary:
#   T(z_max,t) = T0
# ============================================================

import numpy as np
from scipy.special import erfc
from scipy.linalg import solve_banded


# ============================================================
# Mathematical helpers
# ============================================================

def ierfc(x):
    """
    First integral of complementary error function:

        ierfc(x) = exp(-x^2)/sqrt(pi) - x*erfc(x)
    """
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-x**2) / np.sqrt(np.pi) - x * erfc(x)


def solid_preheating_profile(z, mat, I_laser):
    """
    Approximate solid temperature profile at t = t_melt.

    Surface heat flux during preheating:
        -ks dT/dz = A_s I

    Semi-infinite solid solution:
        T(z,t) = T0 + (2 A_s I / ks) sqrt(alpha_s t) ierfc(xi)

    where:
        xi = z / (2 sqrt(alpha_s t))

    Clipped to [T0, Tm], because at t=t_melt the surface should
    not exceed the melting point in this simplified initialization.
    """
    z = np.asarray(z, dtype=np.float64)

    ks = float(mat["ks"])
    alpha_s = float(mat["alpha_s"])
    T0 = float(mat["T0"])
    Tm = float(mat["Tm"])
    A_s = float(mat["A_s"])
    t_melt = float(mat["t_melt"])

    if t_melt <= 0.0:
        return np.full_like(z, T0, dtype=np.float64)

    AI_s = A_s * float(I_laser)
    xi = z / (2.0 * np.sqrt(alpha_s * t_melt) + 1e-30)

    T = T0 + (2.0 * AI_s / ks) * np.sqrt(alpha_s * t_melt) * ierfc(xi)
    return np.clip(T, T0, Tm).astype(np.float64)


def extract_melt_depth(z, T, Tm):
    """
    Extract deepest position where T >= Tm.
    Uses linear interpolation between the last melted node
    and the first solid node.
    """
    z = np.asarray(z, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    idx = np.where(T >= Tm)[0]

    if len(idx) == 0:
        return 0.0

    j = idx[-1]

    if j >= len(z) - 1:
        return float(z[j])

    T1 = T[j]
    T2 = T[j + 1]
    z1 = z[j]
    z2 = z[j + 1]

    if abs(T2 - T1) < 1e-30:
        return float(z1)

    return float(z1 + (Tm - T1) * (z2 - z1) / (T2 - T1))


def add_pre_melting_interval(t, S, t_melt, n_pre=200):
    """
    Add S(t)=0 before the known melting time.
    Used for Ag/Al/Cu/Ti to plot from t=0 to t=t_max.
    """
    if t_melt <= 0.0:
        return t, S

    t_pre = np.linspace(0.0, t_melt, n_pre)
    S_pre = np.zeros_like(t_pre)

    return np.concatenate([t_pre, t]), np.concatenate([S_pre, S])


def final_relative_error(y_pred_final, y_ref_final):
    return abs(y_pred_final - y_ref_final) / (abs(y_ref_final) + 1e-30) * 100.0


def relative_l2_error(y_pred, y_ref):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    return np.linalg.norm(y_pred - y_ref) / (np.linalg.norm(y_ref) + 1e-30)


# ============================================================
# Material coefficient helpers
# ============================================================

def _specific_heats_from_diffusivity(mat):
    """
    Recover c_s and c_l from alpha = k/(rho*c).
    """
    rho_s = float(mat["rho_s"])
    rho_l = float(mat["rho_l"])
    ks = float(mat["ks"])
    kl = float(mat["kl"])
    alpha_s = float(mat["alpha_s"])
    alpha_l = float(mat["alpha_l"])

    cs = ks / (rho_s * alpha_s)
    cl = kl / (rho_l * alpha_l)

    return cs, cl


def _phase_properties(T, mat, dT_mushy):
    """
    Piecewise phase-dependent rho, k, c_eff, alpha_eff.
    """
    T = np.asarray(T, dtype=np.float64)

    rho_s = float(mat["rho_s"])
    rho_l = float(mat["rho_l"])
    ks = float(mat["ks"])
    kl = float(mat["kl"])
    Tm = float(mat["Tm"])
    Lh = float(mat["Lh"])

    cs, cl = _specific_heats_from_diffusivity(mat)

    liquid = T >= Tm
    mushy = (T >= Tm - dT_mushy / 2.0) & (T <= Tm + dT_mushy / 2.0)

    rho = np.where(liquid, rho_l, rho_s)
    k = np.where(liquid, kl, ks)
    c = np.where(liquid, cl, cs)

    c_eff = c.copy()
    c_eff[mushy] += Lh / dT_mushy

    alpha_eff = k / (rho * c_eff)

    return rho, k, c_eff, alpha_eff


def _surface_coefficients(T_surface, mat, force_liquid_surface):
    """
    Select surface absorptivity and conductivity.
    """
    Tm = float(mat["Tm"])

    if force_liquid_surface or T_surface >= Tm:
        return float(mat["A_l"]), float(mat["kl"])

    return float(mat["A_s"]), float(mat["ks"])


def _make_grid_and_time(mat, t_start, t_end, z_max, Nz, dt=None, safety=0.35):
    alpha_s = float(mat["alpha_s"])
    alpha_l = float(mat["alpha_l"])
    alpha_max = max(alpha_s, alpha_l)

    if z_max is None:
        z_max = 15.0 * np.sqrt(alpha_s * max(t_end, 1e-30))

    Nz = int(Nz)
    z = np.linspace(0.0, float(z_max), Nz)
    dz = z[1] - z[0]

    if dt is None:
        dt_cfl = safety * dz**2 / alpha_max
        Nt = int(np.ceil((t_end - t_start) / dt_cfl)) + 1
    else:
        Nt = int(np.ceil((t_end - t_start) / float(dt))) + 1

    Nt = max(Nt, 2)
    t_arr = np.linspace(t_start, t_end, Nt)
    dt_actual = t_arr[1] - t_arr[0]

    return z, dz, t_arr, dt_actual, Nt


# ============================================================
# Explicit FDM solver
# ============================================================

def solve_fdm_enthalpy_explicit(
    mat,
    I_laser,
    t_start=0.0,
    t_end=10.0,
    z_max=None,
    Nz=1000,
    dT_mushy=10.0,
    safety=0.35,
    dt=None,
    T_init=None,
    force_liquid_surface=False,
    save_times=None,
    verbose=True,
):
    """
    Explicit finite-difference enthalpy/effective heat capacity solver.

    Interior:
        T_i^{n+1} = T_i^n + dt * alpha_i^n * (T_{i+1}^n - 2T_i^n + T_{i-1}^n)/dz^2

    Surface heat flux is applied through an explicit ghost-node form:
        -k dT/dz = A I

    This gives:
        T_0^{n+1} = T_0^n + 2 alpha_0 dt/dz^2 * (T_1^n - T_0^n + dz * A I / k)
    """
    Tm = float(mat["Tm"])
    T0 = float(mat["T0"])

    z, dz, t_arr, dt_actual, Nt = _make_grid_and_time(
        mat=mat,
        t_start=t_start,
        t_end=t_end,
        z_max=z_max,
        Nz=Nz,
        dt=dt,
        safety=safety,
    )

    if verbose:
        print(
            f"  [explicit] z_max = {z[-1] * 100:.4f} cm | "
            f"dz = {dz * 1e6:.4f} μm | "
            f"dt = {dt_actual:.3e} s | Nt = {Nt:,}"
        )

    if T_init is None:
        T = np.full(Nt * 0 + len(z), T0, dtype=np.float64)
    else:
        T_init = np.asarray(T_init, dtype=np.float64)
        if len(T_init) != len(z):
            raise ValueError(f"T_init length {len(T_init)} != Nz {len(z)}")
        T = T_init.copy()

    S_arr = np.zeros(Nt, dtype=np.float64)
    save_times = sorted(save_times) if save_times else []
    T_profiles = {}
    save_idx = 0

    for n in range(Nt):
        current_t = t_arr[n]

        while save_idx < len(save_times) and current_t >= save_times[save_idx] - 1e-15:
            T_profiles[save_times[save_idx]] = T.copy()
            save_idx += 1

        S_arr[n] = extract_melt_depth(z, T, Tm)

        if n == Nt - 1:
            break

        _, _, _, alpha_eff = _phase_properties(T, mat, dT_mushy)

        T_new = T.copy()

        # Interior nodes
        T_new[1:-1] = (
            T[1:-1]
            + dt_actual * alpha_eff[1:-1] / dz**2
            * (T[2:] - 2.0 * T[1:-1] + T[:-2])
        )

        # Surface Neumann BC: -k dT/dz = A I
        A_surf, k_surf = _surface_coefficients(T[0], mat, force_liquid_surface)
        flux_term = dz * A_surf * float(I_laser) / k_surf

        T_new[0] = (
            T[0]
            + 2.0 * dt_actual * alpha_eff[0] / dz**2
            * (T[1] - T[0] + flux_term)
        )

        # Far boundary
        T_new[-1] = T0

        # Avoid nonphysical cooling below ambient in this model
        T = np.maximum(T_new, T0)

    return t_arr, S_arr, z, T.copy(), T_profiles


# Backward-compatible alias
solve_fdm_enthalpy = solve_fdm_enthalpy_explicit


# ============================================================
# Implicit FDM solver
# ============================================================

def solve_fdm_enthalpy_implicit(
    mat,
    I_laser,
    t_start=0.0,
    t_end=10.0,
    z_max=None,
    Nz=1000,
    dT_mushy=10.0,
    safety=0.35,
    dt_factor=5.0,
    dt=None,
    T_init=None,
    force_liquid_surface=False,
    save_times=None,
    nonlinear_iters=3,
    tol=1e-8,
    verbose=True,
):
    """
    Implicit finite-difference enthalpy/effective heat capacity solver.

    Linearized implicit scheme:
        coefficients rho, k, c_eff are evaluated from the current iterate.

    Interior:
        -r_i T_{i-1}^{n+1} + (1+2r_i) T_i^{n+1}
        -r_i T_{i+1}^{n+1} = T_i^n

    where:
        r_i = alpha_i dt / dz^2

    Surface heat flux:
        -k dT/dz = A I

    Implemented as an implicit ghost-node condition:
        (1 + 2r_0)T_0^{n+1} - 2r_0 T_1^{n+1}
        = T_0^n + 2r_0 dz A I/k

    Far boundary:
        T_{N-1}^{n+1} = T0

    dt choice:
        If dt is not specified, the solver uses:
            dt = dt_factor * explicit_CFL_dt
        Implicit is more stable, so dt_factor can be > 1.
        For convergence studies use dt_factor = 1, 2, 5, 10.
    """
    Tm = float(mat["Tm"])
    T0 = float(mat["T0"])

    # If dt is not given, use dt_factor times explicit CFL dt.
    if dt is None:
        alpha_max = max(float(mat["alpha_s"]), float(mat["alpha_l"]))

        if z_max is None:
            z_max_local = 15.0 * np.sqrt(float(mat["alpha_s"]) * max(t_end, 1e-30))
        else:
            z_max_local = float(z_max)

        dz_est = z_max_local / (int(Nz) - 1)
        dt_cfl = safety * dz_est**2 / alpha_max
        dt = dt_factor * dt_cfl

    z, dz, t_arr, dt_actual, Nt = _make_grid_and_time(
        mat=mat,
        t_start=t_start,
        t_end=t_end,
        z_max=z_max,
        Nz=Nz,
        dt=dt,
        safety=safety,
    )

    if verbose:
        print(
            f"  [implicit] z_max = {z[-1] * 100:.4f} cm | "
            f"dz = {dz * 1e6:.4f} μm | "
            f"dt = {dt_actual:.3e} s | Nt = {Nt:,} | "
            f"nonlinear_iters = {nonlinear_iters}"
        )

    if T_init is None:
        T = np.full(len(z), T0, dtype=np.float64)
    else:
        T_init = np.asarray(T_init, dtype=np.float64)
        if len(T_init) != len(z):
            raise ValueError(f"T_init length {len(T_init)} != Nz {len(z)}")
        T = T_init.copy()

    S_arr = np.zeros(Nt, dtype=np.float64)
    save_times = sorted(save_times) if save_times else []
    T_profiles = {}
    save_idx = 0

    N = len(z)

    for n in range(Nt):
        current_t = t_arr[n]

        while save_idx < len(save_times) and current_t >= save_times[save_idx] - 1e-15:
            T_profiles[save_times[save_idx]] = T.copy()
            save_idx += 1

        S_arr[n] = extract_melt_depth(z, T, Tm)

        if n == Nt - 1:
            break

        T_old = T.copy()
        T_iter = T.copy()

        for _ in range(int(nonlinear_iters)):
            _, _, _, alpha_eff = _phase_properties(T_iter, mat, dT_mushy)

            A_surf, k_surf = _surface_coefficients(
                T_iter[0],
                mat,
                force_liquid_surface,
            )

            r = alpha_eff * dt_actual / dz**2

            # Banded matrix for solve_banded((1, 1), ab, b)
            # ab[0, 1:] = upper diagonal
            # ab[1, :]  = main diagonal
            # ab[2, :-1] = lower diagonal
            ab = np.zeros((3, N), dtype=np.float64)
            b = T_old.copy()

            # Surface node: (1+2r0)T0 - 2r0*T1 = T_old0 + 2r0*dz*AI/k
            ab[1, 0] = 1.0 + 2.0 * r[0]
            ab[0, 1] = -2.0 * r[0]
            b[0] = T_old[0] + 2.0 * r[0] * dz * A_surf * float(I_laser) / k_surf

            # Interior nodes
            for i in range(1, N - 1):
                ab[2, i - 1] = -r[i]
                ab[1, i] = 1.0 + 2.0 * r[i]
                ab[0, i + 1] = -r[i]

            # Far boundary: T[-1] = T0
            ab[1, N - 1] = 1.0
            b[N - 1] = T0

            T_new = solve_banded((1, 1), ab, b)
            T_new = np.maximum(T_new, T0)

            diff = np.linalg.norm(T_new - T_iter) / (np.linalg.norm(T_iter) + 1e-30)
            T_iter = T_new

            if diff < tol:
                break

        T = T_iter.copy()

    return t_arr, S_arr, z, T.copy(), T_profiles