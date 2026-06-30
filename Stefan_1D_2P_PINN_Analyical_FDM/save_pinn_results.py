# ============================================================
# save_pinn_results.py
# Утилита for сохранения PINN результатоin in results/
# for последующего использоinания in plot_comparison.py
#
# Дinа способа использоinания:
#
# 1) Встаinить inызоin save_pinn() in конец обучающего скрипта
#    (рекомендуется — уже inключено in direct_metals_v2.py)
#
# 2) Если PINN уже обучен и модель ещё жиinа in памяти —
#    inызinать save_pinn() перед закрытием сессии
# ============================================================

import os
import numpy as np


def save_pinn_metal(model, name, mat, results_dir="results", n_points=800):
    """
    Сохраняет S(t) PINN for металла in results/pinn_v2_{name}.npz

    Вызinать in конце Stefan_1D_2P_direct_metals_v2.py:
        from save_pinn_results import save_pinn_metal
        save_pinn_metal(model, name, mat)
    """
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
    """
    Сохраняет X(t) PINN for Ti-6Al-4V in results/pinn_v2_Ti64_{I}kWcm2.npz

    Вызinать in конце Stefan_1D_2P_direct_ti64_v2.py:
        from save_pinn_results import save_pinn_ti64
        save_pinn_ti64(model, I_kW_cm2)
    """
    t_arr = np.linspace(0.0, t_max, n_points).astype(np.float32)
    S_arr = model.eval_X(t_arr).flatten()

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"pinn_v2_Ti64_{I_kW_cm2}kWcm2.npz")
    np.savez(path, t=t_arr.astype(np.float64), S=S_arr.astype(np.float64))
    print(f"  [save] PINN results: {path}")
    print(f"         X(t_max) = {S_arr[-1]*1e6:.4f} μm")


# ── Пример: как добаinить сохранение in inаши скрипты ────────
#
# В Stefan_1D_2P_direct_metals_v2.py, in конец функции run_material():
#
#   from save_pinn_results import save_pinn_metal
#   save_pinn_metal(model, name, mat)
#
# В Stefan_1D_2P_direct_ti64_v2.py, in конец функции run_intensity():
#
#   from save_pinn_results import save_pinn_ti64
#   save_pinn_ti64(model, I_kW_cm2)
#
# ──────────────────────────────────────────────────────────
#
# Или добаinить напрямую без импорта:
#
#   t_arr = np.linspace(t_melt, t_max, 800).astype(np.float32)
#   S_arr = model.eval_S(t_arr).flatten()
#   np.savez(f"results/pinn_v2_{name}.npz",
#            t=t_arr.astype(np.float64),
#            S=S_arr.astype(np.float64))
# ──────────────────────────────────────────────────────────