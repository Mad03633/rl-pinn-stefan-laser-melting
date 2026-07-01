import os
import numpy as np


def save_pinn_metal(model, name, mat, results_dir="results", n_points=800):
    t_melt = float(mat["t_melt"])
    t_max  = float(mat["t_max"])
    t_arr  = np.linspace(t_melt, t_max, n_points).astype(np.float32)
    S_arr  = model.eval_S(t_arr).flatten()

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"pinn_v2_{name}.npz")
    np.savez(path, t=t_arr.astype(np.float64), S=S_arr.astype(np.float64))
    print(f"  [save] PINN results: {path}")
    print(f"         S(t_max) = {S_arr[-1]*100:.4f} cm")


def save_pinn_ti64(model, I_kW_cm2, t_max=7e-6, results_dir="results", n_points=800):
    t_arr = np.linspace(0.0, t_max, n_points).astype(np.float32)
    S_arr = model.eval_X(t_arr).flatten()

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"pinn_v2_Ti64_{I_kW_cm2}kWcm2.npz")
    np.savez(path, t=t_arr.astype(np.float64), S=S_arr.astype(np.float64))
    print(f"  [save] PINN results: {path}")
    print(f"         X(t_max) = {S_arr[-1]*1e6:.4f} μm")