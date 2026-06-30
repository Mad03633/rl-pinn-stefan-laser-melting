# ============================================================
# Stefan_1D_2P_models_metals.py
# PINN for 1D two-phase Stefan — laser melting
# Al, Ag, Cu, Ti  |  I = 1e9 W/m²  |  t in [t_melt, 10 s]
#
# Strategy: identical to Ti-6Al-4V (Wang & Perdikaris + Ngwenya):
#   - Analytical reference (k_S, k_Ts, k_Tl)
#     computed from same quasi-steady formula as Ngwenya
#   - Supervision losses L_data_S, L_data_Ts, L_data_Tl  (always active)
#   - Physics losses (PDE, BC, Stefan) weighted by phys_weight
#   - Curriculum: phys_weight 0 → 0.01 → 1.0
#   - S(t) architecture: S=0 for t <= t_melt enforced analytically
#     via tau = sqrt((t-t_melt)/(t_max-t_melt)) * sigmoid(net)
# ============================================================

import numpy as np
import tensorflow.compat.v1 as tf
from scipy.special import erfc, erfcinv

tf.disable_v2_behavior()


#  Analytical reference  (same quasi-steady as Ngwenya)

def _ierfc(u):
    u = np.asarray(u, dtype=np.float64)
    return np.exp(-u**2) / np.sqrt(np.pi) - u * erfc(u)


def k_S(t_arr, AI_l, ks, alpha_s, Tm, T0, t_melt):
    t_arr = np.asarray(t_arr, dtype=np.float64)
    S = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        tp = t - t_melt
        if tp <= 1e-20:
            continue
        Tsurf = T0 + (2.0 * AI_l / ks) * np.sqrt(alpha_s * tp / np.pi)
        if Tsurf <= Tm:
            continue
        ratio = (Tm - T0) / (Tsurf - T0)
        if 0.0 < ratio < 1.0:
            S[i] = 2.0 * np.sqrt(alpha_s * tp) * erfcinv(ratio)
    return S


def k_Ts(z_arr, t_arr, AI_l, ks, alpha_s, Tm, T0, t_melt):
    z_arr = np.asarray(z_arr, dtype=np.float64)
    t_arr = np.asarray(t_arr, dtype=np.float64)
    T = np.full_like(z_arr, T0)
    for i, (z, t) in enumerate(zip(z_arr, t_arr)):
        tp = t - t_melt
        if tp < 1e-20:
            continue
        xi = z / (2.0 * np.sqrt(alpha_s * tp))
        T[i] = T0 + (2.0 * AI_l / ks) * np.sqrt(alpha_s * tp / np.pi) * _ierfc(xi)
    return np.clip(T, T0, Tm).astype(np.float32)


def k_Tl(z_arr, S_arr, AI_l, kl, Tm):
    z_arr = np.asarray(z_arr, dtype=np.float64)
    S_arr = np.asarray(S_arr, dtype=np.float64)
    return np.maximum(Tm + (AI_l / kl) * (S_arr - z_arr), Tm).astype(np.float32)


#  Network helpers

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


#  PINN class

class StefanMetals:

    def __init__(
        self,
        z_max, t_melt, t_max,
        rho_s, rho_l, ks, kl, alpha_s, alpha_l, Lh, Tm, T0,
        A_l, I,
        S_max_hint=None,
        layers_T=(2, 128, 128, 128, 1),
        layers_S=(1, 128, 128, 128, 1),
        # physics weights
        w_r=1.0, w_ic=10.0, w_bc_l=200.0, w_bc_s=10.0,
        w_xt=800.0, w_xs=80.0, w_x0=10.0, w_xmin=30.0,
        X_min_m=1e-7,
        # supervision weights (always active)
        w_data_S=500.0, w_data_Ts=50.0, w_data_Tl=50.0,
        seed=1234,
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.z_max_f  = float(z_max)
        self.t_melt_f = float(t_melt)
        self.t_max_f  = float(t_max)

        self.AI_l = float(A_l) * float(I)

        # S_max for output scaling
        if S_max_hint is None:
            _t_end = np.array([float(t_max)])
            _S = k_S(_t_end, self.AI_l, ks, alpha_s, Tm, T0, t_melt)
            S_max_hint = float(_S[0]) if _S[0] > 0 else float(z_max) * 0.1
        S_max_hint = float(S_max_hint)
        self.S_max_f = S_max_hint

        # Temperature output ranges
        Tl_surf_max = Tm + (self.AI_l / kl) * S_max_hint
        dT_l_val = max(1.2 * (Tl_surf_max - T0), 1.2 * (Tm - T0))
        dT_s_val = 1.05 * (Tm - T0)
        T_char = float(Tm - T0)

        # PDE scales
        t_dur = float(t_max - t_melt)
        pde_s_val = T_char / t_dur
        pde_l_val = max(
            float(alpha_l) * self.AI_l / (float(kl) * max(S_max_hint, 1e-9)),
            pde_s_val
        )
        q_scale_val = max(self.AI_l, float(kl) * T_char / float(z_max))
        s_scale_val = max(self.AI_l,
                           float(rho_s) * float(Lh) * S_max_hint / t_dur)

        print(f"  [init] AI_l       = {self.AI_l:.3e} W/m²")
        print(f"  [init] S_max_hint = {S_max_hint*100:.3f} cm")
        print(f"  [init] Tl_surf    = {Tl_surf_max:.0f} K  dT_l = {dT_l_val:.0f} K")
        print(f"  [init] pde_l = {pde_l_val:.2e}  pde_s = {pde_s_val:.2e}")
        print(f"  [init] t_melt = {t_melt:.4e} s  t_max = {t_max:.1f} s")

        # TF constants
        C = lambda v: tf.constant(float(v), dtype=tf.float32)
        self.rho_s = C(rho_s); self.rho_l = C(rho_l)
        self.ks = C(ks); self.kl = C(kl)
        self.alpha_s = C(alpha_s); self.alpha_l = C(alpha_l)
        self.Lh = C(Lh)
        self.Tm = C(Tm); self.T0 = C(T0)
        self.AI_tf = C(self.AI_l)
        self.z_max = C(z_max)
        self.t_melt = C(t_melt); self.t_melt_f = float(t_melt)
        self.t_max_tf = C(t_max)
        self.t_span = C(t_max - t_melt)
        self.S_max = C(S_max_hint)
        self.T_char = C(T_char)
        self.dT_l = C(dT_l_val); self.dT_s = C(dT_s_val)
        self.X_min = C(X_min_m)
        self.delta = C(max(1e-3 * S_max_hint, 1e-9))
        self.pde_l = C(pde_l_val); self.pde_s = C(pde_s_val)
        self.q_scale = C(q_scale_val)
        self.s_scale = C(s_scale_val)

        # loss weights
        self.w_r = C(w_r); self.w_ic = C(w_ic)
        self.w_bc_l = C(w_bc_l); self.w_bc_s = C(w_bc_s)
        self.w_xt = C(w_xt); self.w_xs = C(w_xs)
        self.w_x0 = C(w_x0); self.w_xmin = C(w_xmin)
        self.w_data_S = C(w_data_S); self.w_data_Ts = C(w_data_Ts)
        self.w_data_Tl = C(w_data_Tl)

        # networks
        self.net_Tl = FCNN(list(layers_T))
        self.net_Ts = FCNN(list(layers_T))
        self.net_S = FCNN(list(layers_S))

        # placeholders — physics collocation
        self.z_rl = tf.placeholder(tf.float32, [None,1], 'z_rl')
        self.t_rl = tf.placeholder(tf.float32, [None,1], 't_rl')
        self.z_rs = tf.placeholder(tf.float32, [None,1], 'z_rs')
        self.t_rs = tf.placeholder(tf.float32, [None,1], 't_rs')
        self.z_ic = tf.placeholder(tf.float32, [None,1], 'z_ic')
        self.t_bc = tf.placeholder(tf.float32, [None,1], 't_bc')
        self.t_X = tf.placeholder(tf.float32, [None,1], 't_X')

        # placeholders — supervision
        self.t_sup_S = tf.placeholder(tf.float32, [None,1], 't_sup_S')
        self.S_sup = tf.placeholder(tf.float32, [None,1], 'S_sup')
        self.z_sup_Ts = tf.placeholder(tf.float32, [None,1], 'z_sup_Ts')
        self.t_sup_Ts = tf.placeholder(tf.float32, [None,1], 't_sup_Ts')
        self.Ts_sup = tf.placeholder(tf.float32, [None,1], 'Ts_sup')
        self.z_sup_Tl = tf.placeholder(tf.float32, [None,1], 'z_sup_Tl')
        self.t_sup_Tl = tf.placeholder(tf.float32, [None,1], 't_sup_Tl')
        self.Tl_sup = tf.placeholder(tf.float32, [None,1], 'Tl_sup')

        self.lr = tf.placeholder(tf.float32, [], 'lr')
        self.phys_weight = tf.placeholder(tf.float32, [], 'phys_weight')

        self._build_graph()

        gpu_cfg = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)
        gpu_cfg.gpu_options.allow_growth = True
        gpu_cfg.gpu_options.per_process_gpu_memory_fraction = 0.7
        self.sess = tf.Session(config=gpu_cfg)
        self.sess.run(tf.global_variables_initializer())

    # ── normalisation 
    def _nz(self, z):
        return 2.0 * z / self.z_max - 1.0

    def _nt(self, t):
        eps = tf.constant(1e-12, dtype=tf.float32)
        return 2.0 * (t - self.t_melt) / (self.t_span + eps) - 1.0

    def _nzt(self, z, t):
        return tf.concat([self._nz(z), self._nt(t)], axis=1)

    # ── fields
    def S(self, t):
        # S(t_melt)=0 enforced analytically via sqrt(tau)
        # S ~ sqrt(t-t_melt) matches Stefan similarity scaling
        eps = tf.constant(1e-12, dtype=tf.float32)
        tau = tf.clip_by_value(
            (t - self.t_melt) / (self.t_span + eps), 0.0, 1.0
        )
        tau_phys = tf.sqrt(tau + eps)
        return self.S_max * tau_phys * tf.sigmoid(self.net_S(self._nt(t)))

    def Tl(self, z, t):
        s = 0.5 * (tf.tanh(self.net_Tl(self._nzt(z, t))) + 1.0)
        return self.T0 + self.dT_l * s

    def Ts(self, z, t):
        s = 0.5 * (tf.tanh(self.net_Ts(self._nzt(z, t))) + 1.0)
        return self.T0 + self.dT_s * s

    # ── graph
    def _build_graph(self):
        eps = tf.constant(1e-12, dtype=tf.float32)

        # 1. Liquid PDE
        Tl_r = self.Tl(self.z_rl, self.t_rl)
        Tl_z = tf.gradients(Tl_r, self.z_rl)[0]
        self.Lr_l = tf.reduce_mean(tf.square(
            (tf.gradients(Tl_r, self.t_rl)[0]
             - self.alpha_l * tf.gradients(Tl_z, self.z_rl)[0])
            / (self.pde_l + eps)
        ))

        # 2. Solid PDE
        Ts_r = self.Ts(self.z_rs, self.t_rs)
        Ts_z = tf.gradients(Ts_r, self.z_rs)[0]
        self.Lr_s = tf.reduce_mean(tf.square(
            (tf.gradients(Ts_r, self.t_rs)[0]
             - self.alpha_s * tf.gradients(Ts_z, self.z_rs)[0])
            / (self.pde_s + eps)
        ))

        # 3. IC: T(z, t_melt) = T0  (both phases start at ambient)
        t_ic = tf.ones_like(self.z_ic) * self.t_melt
        self.LIC = (
            tf.reduce_mean(tf.square(
                (self.Tl(self.z_ic, t_ic) - self.T0) / (self.T_char + eps)
            )) +
            tf.reduce_mean(tf.square(
                (self.Ts(self.z_ic, t_ic) - self.T0) / (self.T_char + eps)
            ))
        )

        # 4. Surface flux BC: -kl*dTl/dz(0,t) = AI_l
        z_surf = tf.zeros_like(self.t_bc)
        Tl_s = self.Tl(z_surf, self.t_bc)
        self.Lbc_l = tf.reduce_mean(tf.square(
            (-self.kl * tf.gradients(Tl_s, z_surf)[0] - self.AI_tf)
            / (self.q_scale + eps)
        ))

        # 5. Far-field: Ts(z_max, t) = T0
        z_far = tf.ones_like(self.t_bc) * self.z_max
        self.Lbc_s = tf.reduce_mean(tf.square(
            (self.Ts(z_far, self.t_bc) - self.T0) / (self.T_char + eps)
        ))

        # 6. Interface temperature continuity
        S_val = self.S(self.t_X)
        self.LXT = tf.reduce_mean(
            tf.square((self.Tl(S_val, self.t_X) - self.Tm) / (self.dT_l + eps)) +
            tf.square((self.Ts(S_val, self.t_X) - self.Tm) / (self.T_char + eps))
        )

        # 7. Stefan condition
        d = tf.maximum(self.delta, tf.constant(1e-12, dtype=tf.float32))
        z_l = tf.maximum(S_val - d, tf.constant(0.0, dtype=tf.float32))
        z_s = tf.minimum(S_val + d, self.z_max)
        stefan = (self.rho_s * self.Lh * tf.gradients(S_val, self.t_X)[0]
                  - self.ks * tf.gradients(self.Ts(z_s, self.t_X), z_s)[0]
                  + self.kl * tf.gradients(self.Tl(z_l, self.t_X), z_l)[0])
        self.LXS = tf.reduce_mean(
            tf.square(stefan / (self.s_scale + eps))
        )

        # 8. S(t_melt) = 0
        t0_ = tf.ones([1,1], dtype=tf.float32) * self.t_melt
        self.LX0 = tf.reduce_mean(
            tf.square(self.S(t0_) / (self.S_max + eps))
        )

        # 9. Anti-collapse
        self.LXmin = tf.reduce_mean(
            tf.square(tf.nn.relu(self.X_min - S_val) / (self.X_min + eps))
        )

        # 10. Supervision: S(t)
        self.L_data_S = tf.reduce_mean(tf.square(
            (self.S(self.t_sup_S) - self.S_sup) / (self.S_max + eps)
        ))

        # 11. Supervision: Ts(z,t)
        self.L_data_Ts = tf.reduce_mean(tf.square(
            (self.Ts(self.z_sup_Ts, self.t_sup_Ts) - self.Ts_sup)
            / (self.T_char + eps)
        ))

        # 12. Supervision: Tl(z,t)
        self.L_data_Tl = tf.reduce_mean(tf.square(
            (self.Tl(self.z_sup_Tl, self.t_sup_Tl) - self.Tl_sup)
            / (self.dT_l + eps)
        ))

        physics_loss = (
            self.w_r * (self.Lr_l + self.Lr_s) +
            self.w_ic * self.LIC +
            self.w_bc_l * self.Lbc_l +
            self.w_bc_s * self.Lbc_s +
            self.w_xt * self.LXT +
            self.w_xs * self.LXS +
            self.w_x0 * self.LX0 +
            self.w_xmin * self.LXmin
        )

        data_loss = (
            self.w_data_S  * self.L_data_S  +
            self.w_data_Ts * self.L_data_Ts +
            self.w_data_Tl * self.L_data_Tl
        )

        self.loss = self.phys_weight * physics_loss + data_loss
        self.data_loss = data_loss
        self.phys_loss = physics_loss
        self.train_op  = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    # ── training
    def train(self, data, iters=10000, lr=5e-4, print_every=1000,
              phys_weight=1.0):
        for it in range(iters + 1):
            feed = {
                self.z_rl: data['z_rl'], self.t_rl: data['t_rl'],
                self.z_rs: data['z_rs'], self.t_rs: data['t_rs'],
                self.z_ic: data['z_ic'],
                self.t_bc: data['t_bc'],
                self.t_X: data['t_X'],
                self.t_sup_S: data['t_sup_S'], self.S_sup: data['S_sup'],
                self.z_sup_Ts: data['z_sup_Ts'],self.t_sup_Ts: data['t_sup_Ts'],
                self.Ts_sup: data['Ts_sup'],
                self.z_sup_Tl: data['z_sup_Tl'],self.t_sup_Tl: data['t_sup_Tl'],
                self.Tl_sup: data['Tl_sup'],
                self.lr: lr,
                self.phys_weight: float(phys_weight),
            }
            self.sess.run(self.train_op, feed_dict=feed)

            if it % print_every == 0:
                L, Lp, Ld, Ll, Ls, Lic, Lbl, Lbs, Lxt, Lxs, Lx0, Lxm, LdS, LdTs, LdTl = \
                    self.sess.run([
                        self.loss, self.phys_loss, self.data_loss,
                        self.Lr_l, self.Lr_s, self.LIC,
                        self.Lbc_l, self.Lbc_s,
                        self.LXT, self.LXS, self.LX0, self.LXmin,
                        self.L_data_S, self.L_data_Ts, self.L_data_Tl,
                    ], feed_dict=feed)
                print(
                    f"it {it:6d} | loss {L:.3e} [p={Lp:.2e} d={Ld:.2e}] | "
                    f"PDE {Ll:.2e}/{Ls:.2e} | IC {Lic:.2e} | "
                    f"BC {Lbl:.2e}/{Lbs:.2e} | "
                    f"LXT {Lxt:.2e} LXS {Lxs:.2e} | Smin {Lxm:.2e} | "
                    f"[Sup] S={LdS:.2e} Ts={LdTs:.2e} Tl={LdTl:.2e}"
                )

    # ── evaluation
    def eval_S(self, t_np):
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1,1)
        return self.sess.run(self.S(tf.constant(t_np)))

    def eval_Tl(self, z_np, t_np):
        z_np = np.asarray(z_np, dtype=np.float32).reshape(-1,1)
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1,1)
        return self.sess.run(self.Tl(tf.constant(z_np), tf.constant(t_np)))

    def eval_Ts(self, z_np, t_np):
        z_np = np.asarray(z_np, dtype=np.float32).reshape(-1,1)
        t_np = np.asarray(t_np, dtype=np.float32).reshape(-1,1)
        return self.sess.run(self.Ts(tf.constant(z_np), tf.constant(t_np)))