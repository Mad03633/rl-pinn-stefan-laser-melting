# ============================================================
# Stefan_1D_2P_models_ti64_v2.py
# PINN для Ti-6Al-4V — v2
# t = 0…7 мкс, I = 5 / 50 / 500 / 5000 кВт/см²
#
# ИЗМЕНЕНИЯ vs v1 (Stefan_1D_2P_models_tf.py):
#   - Главный эталон: FDM (загружается из results/)
#   - Ngwenya analytical solution is kept only as a reference curve, not as training supervision
#   - Метрика: L2 vs FDM (не только vs Ngwenya)
#   - Нормализация жидкой зоны: z_liq_max = X_scale (как в metals_v2)
#   - Добавлена метрика сравнения FDM vs Ngwenya
# ============================================================

import numpy as np
import tensorflow.compat.v1 as tf
from scipy.special import erfc, erfcinv

tf.disable_v2_behavior()


def _ierfc(u):
    u = np.asarray(u, dtype=np.float64)
    return np.exp(-u**2) / np.sqrt(np.pi) - u * erfc(u)


def ngwenya_X(t_arr, AI, ks, alpha_s, Tm, T0):
    t_arr  = np.asarray(t_arr, dtype=np.float64)
    X      = np.zeros_like(t_arr)
    t_melt = np.pi / (4.0 * alpha_s) * (ks * (Tm - T0) / AI)**2
    for i, t in enumerate(t_arr):
        if t <= t_melt or t < 1e-30:
            continue
        Tsurf = T0 + (2.0 * AI / ks) * np.sqrt(alpha_s * t / np.pi)
        if Tsurf <= Tm:
            continue
        ratio = (Tm - T0) / (Tsurf - T0)
        if 0.0 < ratio < 2.0:
            X[i] = 2.0 * np.sqrt(alpha_s * t) * erfcinv(ratio)
    return X


def ngwenya_Ts(z_arr, t_arr, AI, ks, alpha_s, Tm, T0):
    z_arr = np.asarray(z_arr, dtype=np.float64)
    t_arr = np.asarray(t_arr, dtype=np.float64)
    T = np.full_like(z_arr, T0)
    for i, (z, t) in enumerate(zip(z_arr, t_arr)):
        if t < 1e-30:
            continue
        xi   = z / (2.0 * np.sqrt(alpha_s * t))
        T[i] = T0 + (2.0 * AI / ks) * np.sqrt(alpha_s * t) * _ierfc(xi)
    return np.clip(T, T0, Tm)


def ngwenya_Tl(z_arr, X_arr, AI, kl, Tm):
    z_arr = np.asarray(z_arr, dtype=np.float64)
    X_arr = np.asarray(X_arr, dtype=np.float64)
    return np.maximum(Tm + (AI / kl) * (X_arr - z_arr), Tm)


def load_fdm_ti64(npz_path):
    data = np.load(npz_path)
    return {
        "t_fdm": data["t"].astype(np.float64),
        "S_fdm": data["S"].astype(np.float64),
        "z_fdm": data["z"].astype(np.float64),
        "T_fdm": data["T_final"].astype(np.float64),
    }


def fdm_vs_ngwenya_report(fdm_ref, X_ngwenya, t_max):
    S_fdm = fdm_ref["S_fdm"][-1]
    S_ng  = X_ngwenya[-1]
    diff  = (S_ng - S_fdm) / (S_fdm + 1e-30) * 100
    print(f"\n  ── FDM vs Ngwenya ──────────────────────────")
    print(f"  FDM X(t_max)     = {S_fdm*1e6:.3f} μm")
    print(f"  Ngwenya X(t_max) = {S_ng*1e6:.3f} μm")
    print(f"  Ngwenya overestimates by {diff:.1f}%")


# ─────────────────────────────────────────────────────────
#  Сети
# ─────────────────────────────────────────────────────────

def xavier_init(in_dim, out_dim):
    stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(tf.random_normal([in_dim, out_dim], stddev=stddev, dtype=tf.float32))


class FCNN:
    def __init__(self, layers):
        self.weights, self.biases = [], []
        for l in range(len(layers) - 1):
            self.weights.append(xavier_init(layers[l], layers[l+1]))
            self.biases.append(tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32)))

    def __call__(self, X):
        H = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, w) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]


class Stefan1D2P_v2:

    def __init__(
        self,
        z_min, z_max, t_min, t_max,
        rho, Lh, T0, Tm, ks, kl, alpha_s, alpha_l, A, I,
        layers_T=(2, 128, 128, 128, 1),
        layers_X=(1, 128, 128, 128, 1),
        X_scale=None, I_scale=1000.0, X_max_hint=None,
        fdm_X_max=None,
        # Physics loss weights
        w_r=1.0, w_T0=10.0, w_bc=200.0, w_far=10.0,
        w_xt=800.0, w_xs=80.0, w_x0=10.0, w_xmin=30.0,
        X_min_m=5e-7,
        # Ngwenya supervision weights
        w_data_X=0.0, w_data_Ts=0.0, w_data_Tl=0.0,
        # FDM supervision weights
        w_fdm_X=1000.0, w_fdm_Ts=80.0,
        seed=1234,
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.T0_f      = float(T0)
        self.Tm_f      = float(Tm)
        self.ks_f      = float(ks)
        self.kl_f      = float(kl)
        self.alpha_s_f = float(alpha_s)
        self.alpha_l_f = float(alpha_l)
        self.AI_eff    = float(A) * float(I) * float(I_scale)

        self.z_min = float(z_min); self.z_max_f = float(z_max)
        self.t_min = float(t_min); self.t_max_f = float(t_max)

        # FDM is the main numerical reference, therefore the scaling is based on FDM,
        # not on Ngwenya. This is important at low intensities where Ngwenya may
        # significantly overestimate the melt depth.
        if fdm_X_max is not None:
            X_max_est = float(fdm_X_max)
        elif X_max_hint is not None:
            X_max_est = float(X_max_hint)
        else:
            _X = ngwenya_X(np.array([t_max]), self.AI_eff, ks, alpha_s, Tm, T0)
            X_max_est = float(_X[0])

        Tl_surf_max = Tm + (self.AI_eff / kl) * X_max_est
        dT_l_val    = max(1.2 * (Tl_surf_max - T0), 1.2 * (Tm - T0))
        dT_s_val    = 1.05 * (Tm - T0)
        t_melt_val  = float(np.clip(
            np.pi / (4.0 * alpha_s) * (ks * (Tm - T0) / self.AI_eff)**2,
            0.0, 0.99 * t_max
        ))
        self.t_melt_f = t_melt_val

        # Use a melt-front scale based on FDM, not the full computational domain.
        # z_max is much larger than the physical melt depth, especially at 5 kW/cm².
        # A compact scale improves conditioning of X(t)=X_scale*tau*sigmoid(NN(t)).
        if X_scale is None:
            X_scale = 3.0 * X_max_est
        self.X_scale_f = float(X_scale)

        pde_l_val = max(
            float(alpha_l) * self.AI_eff / (kl * max(X_max_est, 1e-9)),
            float(Tm - T0) / float(t_max)
        )
        pde_s_val = float(Tm - T0) / float(t_max)

        print(f"  [Stefan1D2P] AI_eff={self.AI_eff:.3e} W/m²")
        print(f"  [Stefan1D2P] X_max_est={X_max_est*1e6:.2f} μm  (from FDM if provided)")
        print(f"  [Stefan1D2P] X_scale={float(X_scale)*1e6:.2f} μm")
        print(f"  [Stefan1D2P] dT_l={dT_l_val:.0f} K  t_melt={t_melt_val*1e9:.1f} ns")
        print(f"  [Stefan1D2P] w_fdm_X={w_fdm_X}  w_fdm_Ts={w_fdm_Ts}")
        print(f"  [Stefan1D2P] Ngwenya weights: X={w_data_X}, Ts={w_data_Ts}, Tl={w_data_Tl}")

        C = lambda v: tf.constant(float(v), dtype=tf.float32)
        self.rho     = C(rho);     self.Lh     = C(Lh)
        self.T0      = C(T0);      self.Tm     = C(Tm)
        self.ks      = C(ks);      self.kl     = C(kl)
        self.alpha_s = C(alpha_s); self.alpha_l = C(alpha_l)
        self.AI_tf   = C(self.AI_eff)
        self.X_scale = C(X_scale); self.T_char = C(float(Tm - T0))
        self.dT_s    = C(dT_s_val); self.dT_l  = C(dT_l_val)
        self.X_min   = C(float(X_min_m))
        self.t_melt  = C(t_melt_val)
        self.z_max   = C(float(z_max))
        # Нормализация жидкой зоны по X_scale (не z_max)
        self.z_liq_max = C(float(X_scale))
        self.delta   = C(max(0.01 * float(X_scale), 1e-9))
        self.pde_l   = C(pde_l_val); self.pde_s = C(pde_s_val)

        q0      = max(self.AI_eff, 1.0)
        s_scale = max(self.AI_eff,
                      float(rho) * float(Lh) * float(X_scale) / float(t_max))
        self.q_scale = C(q0); self.s_scale = C(s_scale)

        # Веса
        self.w_r    = C(w_r);    self.w_T0   = C(w_T0)
        self.w_bc   = C(w_bc);   self.w_far  = C(w_far)
        self.w_xt   = C(w_xt);   self.w_xs   = C(w_xs)
        self.w_x0   = C(w_x0);   self.w_xmin = C(w_xmin)
        self.w_data_X  = C(w_data_X)
        self.w_data_Ts = C(w_data_Ts)
        self.w_data_Tl = C(w_data_Tl)
        self.w_fdm_X   = C(w_fdm_X)
        self.w_fdm_Ts  = C(w_fdm_Ts)

        self.net_Tl = FCNN(list(layers_T))
        self.net_Ts = FCNN(list(layers_T))
        self.net_X  = FCNN(list(layers_X))

        # Физика
        self.z_rl  = tf.placeholder(tf.float32, [None, 1], 'z_rl')
        self.t_rl  = tf.placeholder(tf.float32, [None, 1], 't_rl')
        self.z_rs  = tf.placeholder(tf.float32, [None, 1], 'z_rs')
        self.t_rs  = tf.placeholder(tf.float32, [None, 1], 't_rs')
        self.z0    = tf.placeholder(tf.float32, [None, 1], 'z0')
        self.t_bc  = tf.placeholder(tf.float32, [None, 1], 't_bc')
        self.t_X   = tf.placeholder(tf.float32, [None, 1], 't_X')
        # Ngwenya supervision
        self.t_sup_X  = tf.placeholder(tf.float32, [None, 1], 't_sup_X')
        self.X_sup    = tf.placeholder(tf.float32, [None, 1], 'X_sup')
        self.z_sup_Ts = tf.placeholder(tf.float32, [None, 1], 'z_sup_Ts')
        self.t_sup_Ts = tf.placeholder(tf.float32, [None, 1], 't_sup_Ts')
        self.Ts_sup   = tf.placeholder(tf.float32, [None, 1], 'Ts_sup')
        self.z_sup_Tl = tf.placeholder(tf.float32, [None, 1], 'z_sup_Tl')
        self.t_sup_Tl = tf.placeholder(tf.float32, [None, 1], 't_sup_Tl')
        self.Tl_sup   = tf.placeholder(tf.float32, [None, 1], 'Tl_sup')
        # FDM supervision (новые)
        self.t_fdm_X  = tf.placeholder(tf.float32, [None, 1], 't_fdm_X')
        self.X_fdm    = tf.placeholder(tf.float32, [None, 1], 'X_fdm')
        self.z_fdm_Ts = tf.placeholder(tf.float32, [None, 1], 'z_fdm_Ts')
        self.t_fdm_Ts = tf.placeholder(tf.float32, [None, 1], 't_fdm_Ts')
        self.Ts_fdm   = tf.placeholder(tf.float32, [None, 1], 'Ts_fdm')

        self.lr          = tf.placeholder(tf.float32, [], 'lr')
        self.phys_weight = tf.placeholder(tf.float32, [], 'phys_weight')

        self._build_graph()

        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cfg)
        self.sess.run(tf.global_variables_initializer())

    # ── Нормализация ──────────────────────────────────────
    def _norm_z(self, z):
        return 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0

    def _norm_z_liq(self, z):
        """Жидкая зона нормализуется по X_scale"""
        return 2.0 * z / self.z_liq_max - 1.0

    def _norm_t(self, t):
        return 2.0 * (t - self.t_min) / (self.t_max_f - self.t_min) - 1.0

    # ── Поля ─────────────────────────────────────────────
    def X(self, t):
        eps    = tf.constant(1e-12, dtype=tf.float32)
        t_span = tf.constant(self.t_max_f - self.t_melt_f, dtype=tf.float32)
        tau    = tf.clip_by_value((t - self.t_melt) / (t_span + eps), 0.0, 1.0)
        s      = tf.sigmoid(self.net_X(self._norm_t(t)))
        return self.X_scale * tau * s

    def Tl(self, z, t):
        inp = tf.concat([self._norm_z_liq(z), self._norm_t(t)], axis=1)
        s   = 0.5 * (tf.tanh(self.net_Tl(inp)) + 1.0)
        return self.T0 + self.dT_l * s

    def Ts(self, z, t):
        inp = tf.concat([self._norm_z(z), self._norm_t(t)], axis=1)
        s   = 0.5 * (tf.tanh(self.net_Ts(inp)) + 1.0)
        return self.T0 + self.dT_s * s

    def _build_graph(self):
        eps = 1e-12

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

        # 3. IC
        t_zero = tf.zeros_like(self.z0)
        self.LT0 = (
            tf.reduce_mean(tf.square((self.Tl(self.z0, t_zero) - self.T0) / (self.T_char + eps))) +
            tf.reduce_mean(tf.square((self.Ts(self.z0, t_zero) - self.T0) / (self.T_char + eps)))
        )

        # 4. BC surface
        z_surf  = tf.zeros_like(self.t_bc)
        Tl_surf = self.Tl(z_surf, self.t_bc)
        self.Lbc_l = tf.reduce_mean(tf.square(
            (-self.kl * tf.gradients(Tl_surf, z_surf)[0] - self.AI_tf) / (self.q_scale + eps)
        ))

        # 5. BC far
        self.Lbc_s = tf.reduce_mean(tf.square(
            (self.Ts(tf.ones_like(self.t_bc) * self.z_max, self.t_bc) - self.T0)
            / (self.T_char + eps)
        ))

        # 6. T on interface X(t)
        X_val = self.X(self.t_X)
        self.LXT = tf.reduce_mean(
            tf.square((self.Tl(X_val, self.t_X) - self.Tm) / (self.T_char + eps)) +
            tf.square((self.Ts(X_val, self.t_X) - self.Tm) / (self.T_char + eps))
        )

        # 7. Stefan condition on interface X(t)
        d     = tf.maximum(self.delta, tf.constant(1e-9, dtype=tf.float32))
        z_l   = tf.maximum(X_val - d, 0.0)
        z_s   = tf.minimum(X_val + d, self.z_max)
        s_res = (self.rho * self.Lh * tf.gradients(X_val, self.t_X)[0]
                 - self.ks * tf.gradients(self.Ts(z_s, self.t_X), z_s)[0]
                 + self.kl * tf.gradients(self.Tl(z_l, self.t_X), z_l)[0])
        self.LXS = tf.reduce_mean(tf.square(s_res / (self.s_scale + eps)))

        # 8. X(0) = 0
        self.LX0 = tf.reduce_mean(tf.square(
            self.X(tf.zeros([1, 1], dtype=tf.float32)) / (self.X_scale + eps)
        ))

        # 9. Anti-collapse
        self.LXmin = tf.reduce_mean(
            tf.square(tf.nn.relu(self.X_min - X_val) / (self.X_min + eps))
        )

        # 10. Ngwenya X(t) supervision
        self.L_data_X = tf.reduce_mean(tf.square(
            (self.X(self.t_sup_X) - self.X_sup) / (self.X_scale + eps)
        ))

        # 11. Ngwenya Ts supervision
        self.L_data_Ts = tf.reduce_mean(tf.square(
            (self.Ts(self.z_sup_Ts, self.t_sup_Ts) - self.Ts_sup) / (self.T_char + eps)
        ))

        # 12. Ngwenya Tl supervision
        self.L_data_Tl = tf.reduce_mean(tf.square(
            (self.Tl(self.z_sup_Tl, self.t_sup_Tl) - self.Tl_sup) / (self.dT_l + eps)
        ))

        # 13. FDM X(t) supervision (новое)
        self.L_fdm_X = tf.reduce_mean(tf.square(
            (self.X(self.t_fdm_X) - self.X_fdm) / (self.X_scale + eps)
        ))

        # 14. FDM Ts(z, t_max) supervision (новое)
        self.L_fdm_Ts = tf.reduce_mean(tf.square(
            (self.Ts(self.z_fdm_Ts, self.t_fdm_Ts) - self.Ts_fdm) / (self.T_char + eps)
        ))

        physics_loss = (
            self.w_r    * (self.Lr_l + self.Lr_s) +
            self.w_T0   * self.LT0                +
            self.w_bc   * self.Lbc_l              +
            self.w_far  * self.Lbc_s              +
            self.w_xt   * self.LXT                +
            self.w_xs   * self.LXS                +
            self.w_x0   * self.LX0                +
            self.w_xmin * self.LXmin
        )

        ngwenya_loss = (
            self.w_data_X  * self.L_data_X  +
            self.w_data_Ts * self.L_data_Ts +
            self.w_data_Tl * self.L_data_Tl
        )

        fdm_loss = (
            self.w_fdm_X  * self.L_fdm_X  +
            self.w_fdm_Ts * self.L_fdm_Ts
        )

        # Final training loss:
        # FDM is the main reference. Ngwenya loss is computed only for logging,
        # but it is not included in the optimization objective.
        self.loss      = self.phys_weight * physics_loss + fdm_loss
        self.phys_loss = physics_loss
        self.ng_loss   = ngwenya_loss
        self.fdm_sup_loss = fdm_loss
        self.train_op  = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, data, iters=20000, lr=1e-3, print_every=1000, phys_weight=1.0):
        for it in range(iters + 1):
            feed = {
                self.z_rl:      data["z_rl"],     self.t_rl:      data["t_rl"],
                self.z_rs:      data["z_rs"],     self.t_rs:      data["t_rs"],
                self.z0:        data["z0"],
                self.t_bc:      data["t_bc"],
                self.t_X:       data["t_X"],
                self.t_sup_X:   data["t_sup_X"],  self.X_sup:     data["X_sup"],
                self.z_sup_Ts:  data["z_sup_Ts"], self.t_sup_Ts:  data["t_sup_Ts"],
                self.Ts_sup:    data["Ts_sup"],
                self.z_sup_Tl:  data["z_sup_Tl"], self.t_sup_Tl:  data["t_sup_Tl"],
                self.Tl_sup:    data["Tl_sup"],
                self.t_fdm_X:   data["t_fdm_X"],  self.X_fdm:     data["X_fdm"],
                self.z_fdm_Ts:  data["z_fdm_Ts"], self.t_fdm_Ts:  data["t_fdm_Ts"],
                self.Ts_fdm:    data["Ts_fdm"],
                self.lr:            lr,
                self.phys_weight:   float(phys_weight),
            }
            self.sess.run(self.train_op, feed_dict=feed)
            if it % print_every == 0:
                vals = self.sess.run(
                    [self.loss, self.phys_loss, self.ng_loss, self.fdm_sup_loss,
                     self.Lr_l, self.Lr_s, self.LT0, self.Lbc_l, self.Lbc_s,
                     self.LXT, self.LXS, self.LX0, self.LXmin,
                     self.L_data_X, self.L_fdm_X],
                    feed_dict=feed
                )
                L, Lp, Lng, Lf, Ll, Ls, LT, Lbl, Lbs, LXT, LXS, LX0, Lxm, LdX, LfX = vals
                print(
                    f"it {it:6d} | loss {L:.3e} [p={Lp:.2e} ng={Lng:.2e} fdm={Lf:.2e}] | "
                    f"PDE {Ll:.2e}/{Ls:.2e} | IC {LT:.2e} | "
                    f"BC {Lbl:.2e}/{Lbs:.2e} | LXT {LXT:.2e} LXS {LXS:.2e} | "
                    f"[Sup] Ng_X={LdX:.2e} FDM_X={LfX:.2e}"
                )

    def eval_X(self, t_np):
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
        return self.sess.run(self.X(tf.constant(t_np, dtype=tf.float32)))

    def eval_Tl(self, z_np, t_np):
        z_np = np.asarray(z_np, dtype=np.float32).reshape(-1, 1)
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
        return self.sess.run(self.Tl(tf.constant(z_np), tf.constant(t_np)))

    def eval_Ts(self, z_np, t_np):
        z_np = np.asarray(z_np, dtype=np.float32).reshape(-1, 1)
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
        return self.sess.run(self.Ts(tf.constant(z_np), tf.constant(t_np)))

    def compute_fdm_metrics(self, fdm_ref):
        """L2-ошибка PINN vs FDM — главная метрика"""
        t_fdm = fdm_ref["t_fdm"].astype(np.float32)
        S_fdm = fdm_ref["S_fdm"]
        X_pinn = self.eval_X(t_fdm).flatten()
        err_final = abs(X_pinn[-1] - S_fdm[-1]) / (abs(S_fdm[-1]) + 1e-30)
        err_l2    = np.linalg.norm(X_pinn - S_fdm) / (np.linalg.norm(S_fdm) + 1e-30)
        return {
            "X_fdm_final":  S_fdm[-1],
            "X_pinn_final": X_pinn[-1],
            "err_final_%":  err_final * 100,
            "err_l2_%":     err_l2 * 100,
        }