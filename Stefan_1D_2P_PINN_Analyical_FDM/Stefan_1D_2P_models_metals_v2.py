# ============================================================
# Stefan_1D_2P_models_metals_v2.py
# PINN для 1D двухфазной задачи Стефана — лазерное плавление
# Ag, Al, Cu, Ti  |  I = 1e9 W/m²  |  t in [t_melt, 10 s]
#
# ИЗМЕНЕНИЯ ОТНОСИТЕЛЬНО v1:
#   - Главный эталон: FDM (загружается из results/fdm_explicit_*.npz)
#   - Аналитика оставлена только для справки
#   - Добавлены supervision потери: L_sup_S, L_sup_Ts
#   - Нормализация PDE-потерь пересчитана из FDM-данных
#   - Метрика качества: L2-ошибка vs FDM
#   - Правильный масштаб: используется z_liq_max = S_scale для net_Tl
# ============================================================

import numpy as np
import tensorflow.compat.v1 as tf
from scipy.special import erfc as scipy_erfc

tf.disable_v2_behavior()


# ── Вспомогательная функция: ierfc ────────────────────────
def _ierfc(u):
    u = np.asarray(u, dtype=np.float64)
    return np.exp(-u**2) / np.sqrt(np.pi) - u * scipy_erfc(u)


# ── Физически точный IC профиль при t = t_melt ────────────
def preheating_Ts(z_arr, t_melt, A_s, I, ks, alpha_s, Tm, T0):
    z_arr = np.asarray(z_arr, dtype=np.float64)
    AI_s  = float(A_s) * float(I)
    coeff = (2.0 * AI_s / ks) * np.sqrt(alpha_s * t_melt / np.pi)
    xi    = z_arr / (2.0 * np.sqrt(alpha_s * t_melt) + 1e-30)
    return np.clip(T0 + coeff * _ierfc(xi), T0, Tm).astype(np.float32)


# ── Загрузка FDM эталона ───────────────────────────────────
def load_fdm_reference(npz_path):
    data = np.load(npz_path)
    return {
        "t_fdm": data["t"].astype(np.float64),
        "S_fdm": data["S"].astype(np.float64),
        "z_fdm": data["z"].astype(np.float64),
        "T_fdm": data["T_final"].astype(np.float64),
    }


def sample_fdm_supervision(fdm_ref, t_melt, t_max,
                            N_sup_S=3000, N_sup_T=3000, seed=42):
    rng = np.random.RandomState(seed)

    t_fdm = fdm_ref["t_fdm"]
    S_fdm = fdm_ref["S_fdm"]
    z_fdm = fdm_ref["z_fdm"]
    T_fdm = fdm_ref["T_fdm"]

    # Убираем t < t_melt (там S=0, не информативно для PINN)
    mask = t_fdm >= t_melt - 1e-12
    t_valid = t_fdm[mask]
    S_valid = S_fdm[mask]

    # Supervision S(t): случайная выборка из FDM
    idx = rng.choice(len(t_valid), size=min(N_sup_S, len(t_valid)), replace=False)
    t_sup_S = t_valid[idx].astype(np.float32)
    S_sup   = S_valid[idx].astype(np.float32)

    # Supervision Ts(z, t_max): из финального профиля FDM
    # Берём только solid зону: z > S_fdm[-1]
    S_end = S_fdm[-1]
    solid_mask = z_fdm >= S_end * 0.95  # небольшой запас
    z_solid = z_fdm[solid_mask]
    T_solid = T_fdm[solid_mask]

    if len(z_solid) > N_sup_T:
        idx_t = rng.choice(len(z_solid), size=N_sup_T, replace=False)
        z_sup_Ts = z_solid[idx_t].astype(np.float32)
        Ts_sup   = T_solid[idx_t].astype(np.float32)
    else:
        z_sup_Ts = z_solid.astype(np.float32)
        Ts_sup   = T_solid.astype(np.float32)

    # Все supervision по Ts берутся при t = t_max
    t_sup_Ts = np.full(len(z_sup_Ts), t_max, dtype=np.float32)

    return dict(
        t_sup_S=t_sup_S.reshape(-1, 1),
        S_sup=S_sup.reshape(-1, 1),
        z_sup_Ts=z_sup_Ts.reshape(-1, 1),
        t_sup_Ts=t_sup_Ts.reshape(-1, 1),
        Ts_sup=Ts_sup.reshape(-1, 1),
    )


# ── FCNN ──────────────────────────────────────────────────
def xavier_init(in_dim, out_dim):
    stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(
        tf.random_normal([in_dim, out_dim], stddev=stddev, dtype=tf.float32)
    )


class FCNN:
    def __init__(self, layers):
        self.weights, self.biases = [], []
        for i in range(len(layers) - 1):
            self.weights.append(xavier_init(layers[i], layers[i+1]))
            self.biases.append(
                tf.Variable(tf.zeros([1, layers[i+1]], dtype=tf.float32))
            )

    def __call__(self, x):
        H = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, w) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]


# ── Основной класс PINN ───────────────────────────────────
class StefanMetalsV2:

    def __init__(
        self,
        z_max, t_melt, t_max,
        rho_s, rho_l, ks, kl, alpha_s, alpha_l,
        Lh, Tm, T0,
        A_s, A_l, I,
        S_scale=None,
        S_max_hint=None,
        # PDE нормализация (если None — вычисляется из FDM)
        pde_l_scale=None,
        pde_s_scale=None,
        layers_T=(2, 64, 64, 64, 1),
        layers_S=(1, 64, 64, 64, 1),
        # Веса физических потерь
        w_r=1.0, w_ic=50.0, w_bc_l=500.0, w_bc_s=20.0,
        w_xt=800.0, w_xs=100.0, w_x0=20.0, w_xmin=20.0,
        # Веса FDM supervision (новое)
        w_sup_S=300.0, w_sup_Ts=50.0,
        X_min_m=1e-8,
        seed=1234,
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.t_melt_f  = float(t_melt)
        self.t_max_f   = float(t_max)
        self.z_max_f   = float(z_max)

        AI_l = float(A_l) * float(I)
        AI_s = float(A_s) * float(I)

        if S_scale is None:
            S_scale = 5.0 * np.sqrt(float(alpha_s) * float(t_max))
        S_scale = float(S_scale)

        T_char = float(Tm - T0)
        S_for_dTl = float(S_max_hint) if S_max_hint is not None else S_scale
        Tl_surf   = float(Tm) + (AI_l / float(kl)) * S_for_dTl
        dT_l      = max(1.2 * (Tl_surf - float(T0)), 1.2 * T_char)
        dT_s      = 1.05 * T_char
        t_dur     = float(t_max - t_melt)

        # PDE нормализация: предпочтительно из FDM
        if pde_s_scale is None:
            pde_s_scale = T_char / t_dur
        if pde_l_scale is None:
            pde_l_scale = max(
                float(alpha_l) * AI_l / (float(kl) * S_scale),
                pde_s_scale
            )

        q_scale = max(AI_l, float(kl) * T_char / np.sqrt(float(alpha_l) * float(t_max)))
        s_scale = max(AI_l, float(rho_s) * float(Lh) * S_scale / t_dur)

        print(f"  [StefanMetals] S_scale={S_scale*100:.2f}cm  dT_l={dT_l:.0f}K")
        print(f"  [StefanMetals] pde_l={pde_l_scale:.2e}  pde_s={pde_s_scale:.2e}")
        print(f"  [StefanMetals] w_sup_S={w_sup_S}  w_sup_Ts={w_sup_Ts}")

        C = lambda v: tf.constant(float(v), dtype=tf.float32)
        self.rho_s   = C(rho_s);   self.rho_l   = C(rho_l)
        self.ks      = C(ks);      self.kl      = C(kl)
        self.alpha_s = C(alpha_s); self.alpha_l = C(alpha_l)
        self.Lh      = C(Lh)
        self.Tm      = C(Tm);      self.T0      = C(T0)
        self.AI_l    = C(AI_l)
        self.z_max   = C(z_max)
        self.z_liq_max = C(S_scale)
        self.t_melt  = C(t_melt);  self.t_span  = C(t_max - t_melt)
        self.S_scale = C(S_scale)
        self.dT_l    = C(dT_l);    self.dT_s    = C(dT_s)
        self.T_char  = C(T_char)
        self.X_min   = C(X_min_m)
        self.delta   = C(max(1e-3 * S_scale, 1e-9))
        self.pde_l   = C(pde_l_scale); self.pde_s = C(pde_s_scale)
        self.q_scale = C(q_scale); self.s_scale = C(s_scale)

        self.w_r    = C(w_r);    self.w_ic   = C(w_ic)
        self.w_bc_l = C(w_bc_l); self.w_bc_s = C(w_bc_s)
        self.w_xt   = C(w_xt);   self.w_xs   = C(w_xs)
        self.w_x0   = C(w_x0);   self.w_xmin = C(w_xmin)
        self.w_sup_S  = C(w_sup_S)
        self.w_sup_Ts = C(w_sup_Ts)

        self.net_Tl = FCNN(list(layers_T))
        self.net_Ts = FCNN(list(layers_T))
        self.net_S  = FCNN(list(layers_S))

        # Физические плейсхолдеры
        self.z_rl  = tf.placeholder(tf.float32, [None, 1], 'z_rl')
        self.t_rl  = tf.placeholder(tf.float32, [None, 1], 't_rl')
        self.z_rs  = tf.placeholder(tf.float32, [None, 1], 'z_rs')
        self.t_rs  = tf.placeholder(tf.float32, [None, 1], 't_rs')
        self.z_ic  = tf.placeholder(tf.float32, [None, 1], 'z_ic')
        self.Ts_ic = tf.placeholder(tf.float32, [None, 1], 'Ts_ic')
        self.t_bc  = tf.placeholder(tf.float32, [None, 1], 't_bc')
        self.t_S   = tf.placeholder(tf.float32, [None, 1], 't_S')

        # FDM supervision плейсхолдеры (новые)
        self.t_sup_S  = tf.placeholder(tf.float32, [None, 1], 't_sup_S')
        self.S_sup    = tf.placeholder(tf.float32, [None, 1], 'S_sup')
        self.z_sup_Ts = tf.placeholder(tf.float32, [None, 1], 'z_sup_Ts')
        self.t_sup_Ts = tf.placeholder(tf.float32, [None, 1], 't_sup_Ts')
        self.Ts_sup   = tf.placeholder(tf.float32, [None, 1], 'Ts_sup')

        self.lr          = tf.placeholder(tf.float32, [], 'lr')
        self.phys_weight = tf.placeholder(tf.float32, [], 'phys_weight')
        self.sup_weight  = tf.placeholder(tf.float32, [], 'sup_weight')

        self._build_graph()

        gpu_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpu_cfg.gpu_options.allow_growth = True
        gpu_cfg.gpu_options.per_process_gpu_memory_fraction = 0.7
        self.sess = tf.Session(config=gpu_cfg)
        self.sess.run(tf.global_variables_initializer())

    # ── Нормализация ─────────────────────────────────────
    def _nz(self, z):
        return 2.0 * z / self.z_max - 1.0

    def _nz_liq(self, z):
        return 2.0 * z / self.z_liq_max - 1.0

    def _nt(self, t):
        eps = tf.constant(1e-12, dtype=tf.float32)
        return 2.0 * (t - self.t_melt) / (self.t_span + eps) - 1.0

    # ── Выходные поля ─────────────────────────────────────
    def S(self, t):
        eps     = tf.constant(1e-12, dtype=tf.float32)
        tau     = tf.clip_by_value((t - self.t_melt) / (self.t_span + eps), 0.0, 1.0)
        tau_phys = tf.sqrt(tau + eps)
        return self.S_scale * tau_phys * tf.sigmoid(self.net_S(self._nt(t)))

    def Tl(self, z, t):
        inp = tf.concat([self._nz_liq(z), self._nt(t)], axis=1)
        s   = 0.5 * (tf.tanh(self.net_Tl(inp)) + 1.0)
        return self.T0 + self.dT_l * s

    def Ts(self, z, t):
        inp = tf.concat([self._nz(z), self._nt(t)], axis=1)
        s   = 0.5 * (tf.tanh(self.net_Ts(inp)) + 1.0)
        return self.T0 + self.dT_s * s

    # ── Граф потерь ───────────────────────────────────────
    def _build_graph(self):
        eps = tf.constant(1e-12, dtype=tf.float32)

        # 1. Liquid PDE
        Tl_r = self.Tl(self.z_rl, self.t_rl)
        Tl_z = tf.gradients(Tl_r, self.z_rl)[0]
        self.Lr_l = tf.reduce_mean(tf.square(
            (tf.gradients(Tl_r, self.t_rl)[0] - self.alpha_l * tf.gradients(Tl_z, self.z_rl)[0])
            / (self.pde_l + eps)
        ))

        # 2. Solid PDE
        Ts_r = self.Ts(self.z_rs, self.t_rs)
        Ts_z = tf.gradients(Ts_r, self.z_rs)[0]
        self.Lr_s = tf.reduce_mean(tf.square(
            (tf.gradients(Ts_r, self.t_rs)[0] - self.alpha_s * tf.gradients(Ts_z, self.z_rs)[0])
            / (self.pde_s + eps)
        ))

        # 3. IC: Ts(z, t_melt) = preheating (физически точно)
        t_ic = tf.ones_like(self.z_ic) * self.t_melt
        self.LIC = tf.reduce_mean(tf.square(
            (self.Ts(self.z_ic, t_ic) - self.Ts_ic) / (self.T_char + eps)
        ))

        # 4. BC поверхность: -kl * dTl/dz(0,t) = AI_l
        z_surf = tf.zeros_like(self.t_bc)
        Tl_s   = self.Tl(z_surf, self.t_bc)
        self.Lbc_l = tf.reduce_mean(tf.square(
            (-self.kl * tf.gradients(Tl_s, z_surf)[0] - self.AI_l) / (self.q_scale + eps)
        ))

        # 5. BC дальнее поле: Ts(z_max, t) = T0
        z_far = tf.ones_like(self.t_bc) * self.z_max
        self.Lbc_s = tf.reduce_mean(tf.square(
            (self.Ts(z_far, self.t_bc) - self.T0) / (self.T_char + eps)
        ))

        # 6. Условие на интерфейсе: Tl(S,t) = Ts(S,t) = Tm
        S_val = self.S(self.t_S)
        self.LXT = tf.reduce_mean(
            tf.square((self.Tl(S_val, self.t_S) - self.Tm) / (self.T_char + eps)) +
            tf.square((self.Ts(S_val, self.t_S) - self.Tm) / (self.T_char + eps))
        )

        # 7. Условие Стефана: rho_s*Lh*dS/dt = ks*dTs/dz|S - kl*dTl/dz|S
        d   = tf.maximum(self.delta, tf.constant(1e-12, dtype=tf.float32))
        z_l = tf.maximum(S_val - d, tf.constant(0.0, dtype=tf.float32))
        z_s = tf.minimum(S_val + d, self.z_max)
        stefan = (self.rho_s * self.Lh * tf.gradients(S_val, self.t_S)[0]
                  - self.ks * tf.gradients(self.Ts(z_s, self.t_S), z_s)[0]
                  + self.kl * tf.gradients(self.Tl(z_l, self.t_S), z_l)[0])
        self.LXS = tf.reduce_mean(tf.square(stefan / (self.s_scale + eps)))

        # 8. S(t_melt) = 0
        t0_ = tf.ones([1, 1], dtype=tf.float32) * self.t_melt
        self.LX0 = tf.reduce_mean(tf.square(self.S(t0_) / (self.S_scale + eps)))

        # 9. Anti-collapse: S(t) > X_min
        self.LXmin = tf.reduce_mean(
            tf.square(tf.nn.relu(self.X_min - S_val) / (self.X_min + eps))
        )

        # 10. FDM SUPERVISION — S(t) (новое)
        self.L_sup_S = tf.reduce_mean(tf.square(
            (self.S(self.t_sup_S) - self.S_sup) / (self.S_scale + eps)
        ))

        # 11. FDM SUPERVISION — Ts(z, t_max) (новое)
        self.L_sup_Ts = tf.reduce_mean(tf.square(
            (self.Ts(self.z_sup_Ts, self.t_sup_Ts) - self.Ts_sup) / (self.T_char + eps)
        ))

        # Физическая часть потерь
        physics = (
            self.w_r    * (self.Lr_l + self.Lr_s) +
            self.w_ic   * self.LIC                +
            self.w_bc_l * self.Lbc_l              +
            self.w_bc_s * self.Lbc_s              +
            self.w_xt   * self.LXT                +
            self.w_xs   * self.LXS                +
            self.w_x0   * self.LX0                +
            self.w_xmin * self.LXmin
        )

        # FDM supervision часть (всегда активна)
        supervision = (
            self.w_sup_S  * self.L_sup_S  +
            self.w_sup_Ts * self.L_sup_Ts
        )

        # Итоговая потеря:
        # sup_weight управляет балансом supervision в curriculum
        self.loss      = self.phys_weight * physics + self.sup_weight * supervision + self.w_ic * self.LIC
        self.phys_loss = physics
        self.sup_loss  = supervision
        self.train_op  = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, data, iters=10000, lr=5e-4, print_every=1000,
              phys_weight=1.0, sup_weight=1.0):
        for it in range(iters + 1):
            feed = {
                self.z_rl:     data['z_rl'],    self.t_rl:     data['t_rl'],
                self.z_rs:     data['z_rs'],    self.t_rs:     data['t_rs'],
                self.z_ic:     data['z_ic'],    self.Ts_ic:    data['Ts_ic'],
                self.t_bc:     data['t_bc'],    self.t_S:      data['t_S'],
                self.t_sup_S:  data['t_sup_S'], self.S_sup:    data['S_sup'],
                self.z_sup_Ts: data['z_sup_Ts'],self.t_sup_Ts: data['t_sup_Ts'],
                self.Ts_sup:   data['Ts_sup'],
                self.lr:           lr,
                self.phys_weight:  float(phys_weight),
                self.sup_weight:   float(sup_weight),
            }
            self.sess.run(self.train_op, feed_dict=feed)
            if it % print_every == 0:
                vals = self.sess.run(
                    [self.loss, self.phys_loss, self.sup_loss,
                     self.Lr_l, self.Lr_s, self.LIC,
                     self.Lbc_l, self.Lbc_s, self.LXT, self.LXS,
                     self.LX0, self.LXmin, self.L_sup_S, self.L_sup_Ts],
                    feed_dict=feed
                )
                L, Lp, Ls, Ll, Lr, Lic, Lbl, Lbs, Lxt, Lxs, Lx0, Lxm, Lss, Lst = vals
                print(
                    f"it {it:6d} | loss {L:.3e} [p={Lp:.2e} s={Ls:.2e}] | "
                    f"PDE {Ll:.2e}/{Lr:.2e} | IC {Lic:.2e} | "
                    f"BC {Lbl:.2e}/{Lbs:.2e} | LXT {Lxt:.2e} LXS {Lxs:.2e} | "
                    f"[FDM] S={Lss:.2e} Ts={Lst:.2e}"
                )

    def eval_S(self, t_np):
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
        return self.sess.run(self.S(tf.constant(t_np)))

    def eval_Tl(self, z_np, t_np):
        z_np = np.asarray(z_np, dtype=np.float32).reshape(-1, 1)
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
        return self.sess.run(self.Tl(tf.constant(z_np), tf.constant(t_np)))

    def eval_Ts(self, z_np, t_np):
        z_np = np.asarray(z_np, dtype=np.float32).reshape(-1, 1)
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
        return self.sess.run(self.Ts(tf.constant(z_np), tf.constant(t_np)))

    def compute_fdm_metrics(self, fdm_ref, t_melt):
        t_fdm = fdm_ref["t_fdm"]
        S_fdm = fdm_ref["S_fdm"]

        mask = t_fdm >= t_melt - 1e-12
        t_valid = t_fdm[mask].astype(np.float32)
        S_valid = S_fdm[mask]

        S_pinn = self.eval_S(t_valid).flatten()

        err_final = abs(S_pinn[-1] - S_valid[-1]) / (abs(S_valid[-1]) + 1e-30)
        err_l2    = np.linalg.norm(S_pinn - S_valid) / (np.linalg.norm(S_valid) + 1e-30)
        err_max   = np.max(np.abs(S_pinn - S_valid))

        return {
            "S_fdm_final":  S_valid[-1],
            "S_pinn_final": S_pinn[-1],
            "err_final_%":  err_final * 100,
            "err_l2_%":     err_l2 * 100,
            "err_max_m":    err_max,
        }