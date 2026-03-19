# ============================================================
# Stefan_1D_2P_models_metals.py
# PINN for 1D two-phase Stefan — laser melting
# Al, Ag, Cu, Ti  |  I = 1e9 W/m²  |  t in [t_melt, 10 s]
#
# Strategy: pure physics — NO supervision on S(t)
#   Goal: PINN solves full PDEs, the result is compared with the analysis
# Supervision only on IC: Ts(z,t_melt) = preheating_Ts(...)
# This is a physical fact, not an analytical approximation of S(t)
#
# Fix vs v3 (BC stuck at 1.00):
# - z_rl is sampled to [0, sqrt(alpha_l*(t-t_melt))] for each point
# (actual scale of the liquid zone), not [0, S_scale=20cm]
# - Additional points near z=0 where BC should be performed
# - w_bc_l increased, BC normalization corrected
# - phys_weight curriculum: 0 (IC only) -> 0.01 -> 1.0
# ============================================================

import numpy as np
import tensorflow.compat.v1 as tf
from scipy.special import erfc as scipy_erfc

tf.disable_v2_behavior()


# ── Physically correct IC: warm-up profile to t_melt ──────
def _ierfc(u):
    u = np.asarray(u, dtype=np.float64)
    return np.exp(-u**2) / np.sqrt(np.pi) - u * scipy_erfc(u)

def preheating_Ts(z_arr, t_melt, A_s, I, ks, alpha_s, Tm, T0):
    """
    Ts(z, t_melt) = T0 + (2*A_s*I/ks)*sqrt(alpha_s*t_melt/pi)*ierfc(xi)
    This is an exact analytical result for a half-space with constant flux.
    """
    z_arr = np.asarray(z_arr, dtype=np.float64)
    AI_s  = float(A_s) * float(I)
    coeff = (2.0 * AI_s / ks) * np.sqrt(alpha_s * t_melt / np.pi)
    xi    = z_arr / (2.0 * np.sqrt(alpha_s * t_melt) + 1e-30)
    return np.clip(T0 + coeff * _ierfc(xi), T0, Tm).astype(np.float32)


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


class StefanMetals:
    """
    Pure-physics PINN for the Stefan problem after melting begins.

    Supervision is ONLY on the IC (physically accurate heating profile).
    S(t) is obtained from physics, without supervision on the analytical curve.
    """

    def __init__(
        self,
        z_max, t_melt, t_max,
        rho_s, rho_l, ks, kl, alpha_s, alpha_l,
        Lh, Tm, T0,
        A_s, A_l, I,
        S_scale=None,
        S_max_hint=None,    
        layers_T=(2, 128, 128, 128, 1),
        layers_S=(1, 128, 128, 128, 1),
        w_r=1.0, w_ic=50.0, w_bc_l=500.0, w_bc_s=20.0,
        w_xt=800.0, w_xs=100.0, w_x0=20.0, w_xmin=20.0,
        X_min_m=1e-8,
        seed=1234,
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.t_melt_f  = float(t_melt)
        self.t_max_f   = float(t_max)
        self.z_max_f   = float(z_max)
        self.alpha_l_f = float(alpha_l)
        self.alpha_s_f = float(alpha_s)

        AI_l = float(A_l) * float(I)
        AI_s = float(A_s) * float(I)

        if S_scale is None:
            S_scale = 5.0 * np.sqrt(float(alpha_s) * float(t_max))
        S_scale = float(S_scale)

        T_char   = float(Tm - T0)
        # dT_l from the actual S_max (S_max_hint), not from S_scale
        # S_scale is too high -> dT_l is unrealistically large -> the network cannot learn
        # Boiling cap is NOT needed: large dT_l is physically correct and
        # is necessary for the network to represent the BC gradient
        S_for_dTl  = float(S_max_hint) if S_max_hint is not None else S_scale
        Tl_surf    = float(Tm) + (AI_l / float(kl)) * S_for_dTl
        dT_l       = max(1.2 * (Tl_surf - float(T0)), 1.2 * T_char)
        dT_s     = 1.05 * T_char
        t_dur    = float(t_max - t_melt)

        pde_s    = T_char / t_dur
        pde_l    = max(float(alpha_l) * AI_l / (float(kl) * S_scale), pde_s)

        # BC normalization: we use kl*(Tm-T0)/sqrt(alpha_l*t_max)
        # — characteristic temperature gradient in the liquid
        q_scale  = max(AI_l, float(kl) * T_char / np.sqrt(float(alpha_l) * float(t_max)))
        s_scale  = max(AI_l, float(rho_s) * float(Lh) * S_scale / t_dur)

        Ts_surf = preheating_Ts(np.array([0.0]), t_melt, A_s, I, ks, alpha_s, Tm, T0)[0]

        print(f"  AI_l = {AI_l:.3e} W/m²  AI_s = {AI_s:.3e} W/m²")
        print(f"  S_scale = {S_scale*100:.2f} cm")
        print(f"  dT_l = {dT_l:.0f} K  dT_s = {dT_s:.0f} K")
        print(f"  pde_l = {pde_l:.2e}  pde_s = {pde_s:.2e}")
        print(f"  q_scale = {q_scale:.2e}  s_scale = {s_scale:.2e}")
        print(f"  Ts(0, t_melt) = {Ts_surf:.0f} K  (IC, аналитически точно)")

        C = lambda v: tf.constant(float(v), dtype=tf.float32)
        self.rho_s   = C(rho_s);   self.rho_l   = C(rho_l)
        self.ks      = C(ks);      self.kl      = C(kl)
        self.alpha_s = C(alpha_s); self.alpha_l = C(alpha_l)
        self.Lh      = C(Lh)
        self.Tm      = C(Tm);      self.T0      = C(T0)
        self.AI_l    = C(AI_l)
        self.z_max      = C(z_max)
        # z_liq_max = S_scale: normalize net_Tl to the maximum liquid zone
        # CRITICAL: z_rl is sampled to S_scale -> normalization must also be S_scale
        # If z_liq_max << S_scale: nz = 2*z/z_liq_max-1 >> 1 -> tanh is saturated -> grad = 0
        # If z_liq_max = S_scale: nz in [-1,1] for all sampling points
        z_liq_max       = S_scale  # = S_scale, not sqrt(alpha_l*t_max)
        self.z_liq_max  = C(z_liq_max)
        print(f'  z_liq_max = {z_liq_max*100:.2f} cm  (= S_scale, liquid normalisation scale)')
        self.t_melt  = C(t_melt);  self.t_span  = C(t_max - t_melt)
        self.S_scale = C(S_scale)
        self.dT_l    = C(dT_l);    self.dT_s    = C(dT_s)
        self.T_char  = C(T_char)
        self.X_min   = C(X_min_m)
        self.delta   = C(max(1e-3 * S_scale, 1e-9))
        self.pde_l   = C(pde_l);   self.pde_s   = C(pde_s)
        self.q_scale = C(q_scale); self.s_scale = C(s_scale)

        self.w_r    = C(w_r);    self.w_ic   = C(w_ic)
        self.w_bc_l = C(w_bc_l); self.w_bc_s = C(w_bc_s)
        self.w_xt   = C(w_xt);   self.w_xs   = C(w_xs)
        self.w_x0   = C(w_x0);   self.w_xmin = C(w_xmin)

        self.net_Tl = FCNN(list(layers_T))
        self.net_Ts = FCNN(list(layers_T))
        self.net_S  = FCNN(list(layers_S))

        # physics placeholders
        self.z_rl  = tf.placeholder(tf.float32, [None,1], 'z_rl')
        self.t_rl  = tf.placeholder(tf.float32, [None,1], 't_rl')
        self.z_rs  = tf.placeholder(tf.float32, [None,1], 'z_rs')
        self.t_rs  = tf.placeholder(tf.float32, [None,1], 't_rs')
        self.t_bc  = tf.placeholder(tf.float32, [None,1], 't_bc')
        self.t_S   = tf.placeholder(tf.float32, [None,1], 't_S')
        # IC placeholder (Ts supervised, физически точный)
        self.z_ic  = tf.placeholder(tf.float32, [None,1], 'z_ic')
        self.Ts_ic = tf.placeholder(tf.float32, [None,1], 'Ts_ic')

        self.lr          = tf.placeholder(tf.float32, [], 'lr')
        self.phys_weight = tf.placeholder(tf.float32, [], 'phys_weight')

        self._build_graph()

        gpu_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpu_cfg.gpu_options.allow_growth = True
        gpu_cfg.gpu_options.per_process_gpu_memory_fraction = 0.7
        self.sess = tf.Session(config=gpu_cfg)
        self.sess.run(tf.global_variables_initializer())

    def _nz(self, z):
        return 2.0 * z / self.z_max - 1.0

    def _nz_liq(self, z):
        """Normalization for the liquid zone by z_liq_max (not z_max)."""
        return 2.0 * z / self.z_liq_max - 1.0

    def _nt(self, t):
        eps = tf.constant(1e-12, dtype=tf.float32)
        return 2.0 * (t - self.t_melt) / (self.t_span + eps) - 1.0

    def _nzt(self, z, t):
        return tf.concat([self._nz(z), self._nt(t)], axis=1)

    def _nzt_liq(self, z, t):
        """Inputs for net_Tl: z is normalized by z_liq_max."""
        return tf.concat([self._nz_liq(z), self._nt(t)], axis=1)

    def S(self, t):
        eps      = tf.constant(1e-12, dtype=tf.float32)
        tau      = tf.clip_by_value((t - self.t_melt) / (self.t_span + eps), 0.0, 1.0)
        tau_phys = tf.sqrt(tau + eps)   # S ~ sqrt(t-t_melt): Stefan's physical scale
        return self.S_scale * tau_phys * tf.sigmoid(self.net_S(self._nt(t)))

    def Tl(self, z, t):
        s = 0.5 * (tf.tanh(self.net_Tl(self._nzt_liq(z, t))) + 1.0)
        return self.T0 + self.dT_l * s

    def Ts(self, z, t):
        s = 0.5 * (tf.tanh(self.net_Ts(self._nzt(z, t))) + 1.0)
        return self.T0 + self.dT_s * s

    def _build_graph(self):
        eps = tf.constant(1e-12, dtype=tf.float32)

        # 1. Liquid PDE: dTl/dt = alpha_l * d2Tl/dz2
        Tl_r  = self.Tl(self.z_rl, self.t_rl)
        Tl_z  = tf.gradients(Tl_r, self.z_rl)[0]
        self.Lr_l = tf.reduce_mean(tf.square(
            (tf.gradients(Tl_r, self.t_rl)[0] - self.alpha_l * tf.gradients(Tl_z, self.z_rl)[0])
            / (self.pde_l + eps)
        ))

        # 2. Solid PDE: dTs/dt = alpha_s * d2Ts/dz2
        Ts_r  = self.Ts(self.z_rs, self.t_rs)
        Ts_z  = tf.gradients(Ts_r, self.z_rs)[0]
        self.Lr_s = tf.reduce_mean(tf.square(
            (tf.gradients(Ts_r, self.t_rs)[0] - self.alpha_s * tf.gradients(Ts_z, self.z_rs)[0])
            / (self.pde_s + eps)
        ))

        # 3. IC: Ts(z, t_melt) = preheating_Ts (physically accurate profile)
        t_ic = tf.ones_like(self.z_ic) * self.t_melt
        self.LIC = tf.reduce_mean(tf.square(
            (self.Ts(self.z_ic, t_ic) - self.Ts_ic) / (self.T_char + eps)
        ))

        # 4. BC surface: -kl * dTl/dz(0,t) = AI_l
        z_surf = tf.zeros_like(self.t_bc)
        Tl_s   = self.Tl(z_surf, self.t_bc)
        self.Lbc_l = tf.reduce_mean(tf.square(
            (-self.kl * tf.gradients(Tl_s, z_surf)[0] - self.AI_l) / (self.q_scale + eps)
        ))

        # 5. BC far field: Ts(z_max, t) = T0
        z_far = tf.ones_like(self.t_bc) * self.z_max
        self.Lbc_s = tf.reduce_mean(tf.square(
            (self.Ts(z_far, self.t_bc) - self.T0) / (self.T_char + eps)
        ))

        # 6. Condition on the interface: Tl(S,t) = Ts(S,t) = Tm
        S_val = self.S(self.t_S)
        # LXT: Both conditions are normalized to T_char (not dT_l!)
        # dT_l >> T_char -> normalizing to dT_l reduces the penalty by a factor of dT_l/T_char
        self.LXT = tf.reduce_mean(
            tf.square((self.Tl(S_val, self.t_S) - self.Tm) / (self.T_char + eps)) +
            tf.square((self.Ts(S_val, self.t_S) - self.Tm) / (self.T_char + eps))
        )

        # 7. Stephan condition: rho_s*Lh*dS/dt = ks*dTs/dz|S - kl*dTl/dz|S
        d   = tf.maximum(self.delta, tf.constant(1e-12, dtype=tf.float32))
        z_l = tf.maximum(S_val - d, tf.constant(0.0, dtype=tf.float32))
        z_s = tf.minimum(S_val + d, self.z_max)
        stefan = (self.rho_s * self.Lh * tf.gradients(S_val, self.t_S)[0]
                  - self.ks * tf.gradients(self.Ts(z_s, self.t_S), z_s)[0]
                  + self.kl * tf.gradients(self.Tl(z_l, self.t_S), z_l)[0])
        self.LXS = tf.reduce_mean(tf.square(stefan / (self.s_scale + eps)))

        # 8. S(t_melt) = 0
        t0_ = tf.ones([1,1], dtype=tf.float32) * self.t_melt
        self.LX0 = tf.reduce_mean(tf.square(self.S(t0_) / (self.S_scale + eps)))

        # 9. Anti-collapse: S(t) > X_min
        self.LXmin = tf.reduce_mean(
            tf.square(tf.nn.relu(self.X_min - S_val) / (self.X_min + eps))
        )

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

        # IC always active (phys_weight does not affect IC)
        self.loss      = self.phys_weight * physics + self.w_ic * self.LIC
        self.phys_loss = physics
        self.train_op  = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, data, iters=10000, lr=5e-4, print_every=1000, phys_weight=1.0):
        for it in range(iters + 1):
            feed = {
                self.z_rl:  data['z_rl'],  self.t_rl:  data['t_rl'],
                self.z_rs:  data['z_rs'],  self.t_rs:  data['t_rs'],
                self.z_ic:  data['z_ic'],  self.Ts_ic: data['Ts_ic'],
                self.t_bc:  data['t_bc'],
                self.t_S:   data['t_S'],
                self.lr:          lr,
                self.phys_weight: float(phys_weight),
            }
            self.sess.run(self.train_op, feed_dict=feed)
            if it % print_every == 0:
                L, Lp, Ll, Ls, Lic, Lbl, Lbs, Lxt, Lxs, Lx0, Lxm = \
                    self.sess.run([self.loss, self.phys_loss,
                                   self.Lr_l, self.Lr_s, self.LIC,
                                   self.Lbc_l, self.Lbc_s,
                                   self.LXT, self.LXS, self.LX0, self.LXmin],
                                  feed_dict=feed)
                print(f"it {it:6d} | loss {L:.3e} [p={Lp:.2e}] | "
                      f"PDE {Ll:.2e}/{Ls:.2e} | IC {Lic:.2e} | "
                      f"BC {Lbl:.2e}/{Lbs:.2e} | "
                      f"LXT {Lxt:.2e} LXS {Lxs:.2e} | "
                      f"S0 {Lx0:.2e} Smin {Lxm:.2e}")

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