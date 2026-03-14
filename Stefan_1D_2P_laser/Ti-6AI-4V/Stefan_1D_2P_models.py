# Stefan_1D_2P_models.py
# PINN for 1D two-phase Stefan problem — laser melting Ti-6Al-4V

import numpy as np
import tensorflow.compat.v1 as tf
from scipy.special import erfc, erfcinv

tf.disable_v2_behavior()

#  Analytical reference  (NumPy, no TF)

def _ierfc(u):
    u = np.asarray(u, dtype=np.float64)
    return np.exp(-u**2) / np.sqrt(np.pi) - u * erfc(u)


def ngwenya_X(t_arr, AI, ks, alpha_s, Tm, T0):
    t_arr = np.asarray(t_arr, dtype=np.float64)
    X = np.zeros_like(t_arr)
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
        xi = z / (2.0 * np.sqrt(alpha_s * t))
        T[i] = T0 + (2.0 * AI / ks) * np.sqrt(alpha_s * t) * _ierfc(xi)
    return np.clip(T, T0, Tm)


def ngwenya_Tl(z_arr, X_arr, AI, kl, Tm):
    z_arr = np.asarray(z_arr, dtype=np.float64)
    X_arr = np.asarray(X_arr, dtype=np.float64)
    return np.maximum(Tm + (AI / kl) * (X_arr - z_arr), Tm)

#  Network helpers

def xavier_init(in_dim, out_dim):
    stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(tf.random_normal([in_dim, out_dim], stddev=stddev, dtype=tf.float32))


class FCNN(object):
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


#  PINN class

class Stefan1D2P(object):
    def __init__(
        self,
        z_min, z_max, t_min, t_max,
        rho, Lh, T0, Tm, ks, kl, alpha_s, alpha_l, A, I,
        layers_T=(2, 100, 100, 100, 1),
        layers_X=(1, 100, 100, 100, 1),
        X_scale=None,
        seed=1234,
        # physics weights
        w_r=1.0, w_T0=10.0, w_bc=200.0, w_far=10.0,
        w_xt=800.0, w_xs=80.0, w_x0=10.0, w_xmin=30.0,
        X_min_m=5e-7,
        # Ngwenya supervision weights
        w_data_X=500.0, w_data_Ts=50.0, w_data_Tl=50.0,
        # effective flux = A * I * I_scale
        I_scale=1000.0,
        # X_max hint for dT_l (if None, computed from AI and t_max)
        X_max_hint=None,
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.T0_f = float(T0)
        self.Tm_f = float(Tm)
        self.ks_f = float(ks)
        self.kl_f = float(kl)
        self.alpha_s_f = float(alpha_s)
        self.alpha_l_f = float(alpha_l)
        self.A_f = float(A)
        self.I_f = float(I)
        self.I_scale_f = float(I_scale)
        self.AI_eff = self.A_f * self.I_f * self.I_scale_f  # W/m^2

        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        # Compute dT_l from actual Tl_surf_max
        # Tl(z=0, t) = Tm + (AI/kl)*X(t) -> max at t=t_max
        if X_max_hint is None:
            # quick estimate: compute analytical X at t_max
            _X_arr = ngwenya_X(
                np.array([t_max]), self.AI_eff, ks, alpha_s, Tm, T0
            )
            X_max_est = float(_X_arr[0])
        else:
            X_max_est = float(X_max_hint)
        # max liquid surface temperature
        Tl_surf_max = Tm + (self.AI_eff / kl) * X_max_est
        dT_l_val = max(1.2 * (Tl_surf_max - T0), 1.2 * (Tm - T0))
        dT_s_val = 1.05 * (Tm - T0) # solid never exceeds Tm

        # Melting onset time (analytical) — encoded in X(t) architecture
        # Before t_melt the surface hasn't reached Tm, so X=0 exactly
        t_melt_val = np.pi / (4.0 * alpha_s) * (ks * (Tm - T0) / self.AI_eff)**2
        # Clamp to valid range: must be < t_max, >= 0
        t_melt_val = float(np.clip(t_melt_val, 0.0, 0.99 * t_max))

        print(f"  [PINN init] AI_eff = {self.AI_eff:.3e} W/m^2")
        print(f"  [PINN init] X_max_est = {X_max_est*1e6:.2f} µm")
        print(f"  [PINN init] Tl_surf_max = {Tl_surf_max:.0f} K")
        print(f"  [PINN init] dT_l = {dT_l_val:.0f} K  |  dT_s = {dT_s_val:.0f} K")

        self.delta  = tf.constant(1e-4 * self.z_max, dtype=tf.float32)

        # TF constants
        self.rho = tf.constant(float(rho), dtype=tf.float32)
        self.Lh = tf.constant(float(Lh), dtype=tf.float32)
        self.T0 = tf.constant(float(T0), dtype=tf.float32)
        self.Tm = tf.constant(float(Tm), dtype=tf.float32)
        self.ks = tf.constant(float(ks), dtype=tf.float32)
        self.kl = tf.constant(float(kl), dtype=tf.float32)
        self.alpha_s = tf.constant(float(alpha_s), dtype=tf.float32)
        self.alpha_l = tf.constant(float(alpha_l), dtype=tf.float32)
        self.AI_tf = tf.constant(self.AI_eff, dtype=tf.float32)

        if X_scale is None:
            X_scale = self.z_max
        self.X_scale = tf.constant(float(X_scale), dtype=tf.float32)
        self.T_char = tf.constant(float(Tm - T0), dtype=tf.float32)
        self.dT_s = tf.constant(dT_s_val, dtype=tf.float32)
        self.dT_l = tf.constant(dT_l_val, dtype=tf.float32)
        self.X_min = tf.constant(float(X_min_m), dtype=tf.float32)
        # t_melt: melting onset — X(t)=0 analytically for t <= t_melt
        self.t_melt  = tf.constant(t_melt_val, dtype=tf.float32)
        self.t_melt_f = t_melt_val

        # PDE scale for liquid: use actual gradient scale
        # alpha_l * d2Tl/dz2 ~ alpha_l * (AI/kl) / X_max
        pde_l_val = max(
            float(alpha_l) * self.AI_eff / (kl * max(X_max_est, 1e-9)),
            float(Tm - T0) / float(t_max)
        )
        pde_s_val = float(Tm - T0) / float(t_max)
        self.pde_scale_l = tf.constant(pde_l_val, dtype=tf.float32)
        self.pde_scale_s = tf.constant(pde_s_val, dtype=tf.float32)
        print(f"  [PINN init] pde_scale_l = {pde_l_val:.2e} K/s  pde_scale_s = {pde_s_val:.2e} K/s")
        print(f"  [PINN init] t_melt = {t_melt_val*1e9:.3f} ns  (X=0 enforced analytically for t <= t_melt)")

        # loss weights
        self.w_r = tf.constant(float(w_r), dtype=tf.float32)
        self.w_T0 = tf.constant(float(w_T0), dtype=tf.float32)
        self.w_bc = tf.constant(float(w_bc), dtype=tf.float32)
        self.w_far = tf.constant(float(w_far), dtype=tf.float32)
        self.w_xt = tf.constant(float(w_xt), dtype=tf.float32)
        self.w_xs = tf.constant(float(w_xs), dtype=tf.float32)
        self.w_x0 = tf.constant(float(w_x0), dtype=tf.float32)
        self.w_xmin = tf.constant(float(w_xmin), dtype=tf.float32)
        self.w_data_X = tf.constant(float(w_data_X), dtype=tf.float32)
        self.w_data_Ts = tf.constant(float(w_data_Ts), dtype=tf.float32)
        self.w_data_Tl = tf.constant(float(w_data_Tl), dtype=tf.float32)

        # networks
        self.net_Tl = FCNN(list(layers_T))
        self.net_Ts = FCNN(list(layers_T))
        self.net_X  = FCNN(list(layers_X))

        # placeholders: physics
        self.z_rl = tf.placeholder(tf.float32, [None, 1], name='z_rl')
        self.t_rl = tf.placeholder(tf.float32, [None, 1], name='t_rl')
        self.z_rs = tf.placeholder(tf.float32, [None, 1], name='z_rs')
        self.t_rs = tf.placeholder(tf.float32, [None, 1], name='t_rs')
        self.z0 = tf.placeholder(tf.float32, [None, 1], name='z0')
        self.t_bc = tf.placeholder(tf.float32, [None, 1], name='t_bc')
        self.t_X = tf.placeholder(tf.float32, [None, 1], name='t_X')

        # placeholders: Ngwenya supervision
        self.t_sup_X = tf.placeholder(tf.float32, [None, 1], name='t_sup_X')
        self.X_sup = tf.placeholder(tf.float32, [None, 1], name='X_sup')
        self.z_sup_Ts = tf.placeholder(tf.float32, [None, 1], name='z_sup_Ts')
        self.t_sup_Ts = tf.placeholder(tf.float32, [None, 1], name='t_sup_Ts')
        self.Ts_sup = tf.placeholder(tf.float32, [None, 1], name='Ts_sup')
        self.z_sup_Tl = tf.placeholder(tf.float32, [None, 1], name='z_sup_Tl')
        self.t_sup_Tl = tf.placeholder(tf.float32, [None, 1], name='t_sup_Tl')
        self.Tl_sup = tf.placeholder(tf.float32, [None, 1], name='Tl_sup')

        self.lr = tf.placeholder(tf.float32, [], name='lr')
        # curriculum flag: when 1.0 physics losses are ON, when 0.0 data only
        self.phys_weight = tf.placeholder(tf.float32, [], name='phys_weight')

        self._build_graph()

        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        )
        self.sess.run(tf.global_variables_initializer())

    # normalisation
    def _norm_z(self, z):
        return 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0

    def _norm_t(self, t):
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def _norm_zt(self, z, t):
        return tf.concat([self._norm_z(z), self._norm_t(t)], axis=1)

    # fields
    def X(self, t):
        # Analytically encode melting onset:
        #   X(t) = 0                              for t <= t_melt
        #   X(t) = X_scale * tau * sigmoid(net)   for t >  t_melt
        # where tau = (t - t_melt) / (t_max - t_melt) in [0, 1]
        # This matches the analytical solution: surface must first reach Tm before any melting occurs.
        eps   = tf.constant(1e-12, dtype=tf.float32)
        t_span = tf.constant(self.t_max - self.t_melt_f, dtype=tf.float32)
        tau   = tf.clip_by_value((t - self.t_melt) / (t_span + eps), 0.0, 1.0)
        s     = tf.sigmoid(self.net_X(self._norm_t(t)))
        return self.X_scale * tau * s

    def Tl(self, z, t):
        # Liquid: T0 + dT_l * sigmoid — can reach Tl_surf_max
        s = 0.5 * (tf.tanh(self.net_Tl(self._norm_zt(z, t))) + 1.0)
        return self.T0 + self.dT_l * s

    def Ts(self, z, t):
        # Solid: T0 + dT_s * sigmoid — bounded at Tm
        s = 0.5 * (tf.tanh(self.net_Ts(self._norm_zt(z, t))) + 1.0)
        return self.T0 + self.dT_s * s

    def _build_graph(self):
        eps = 1e-12
        Tchar = self.T_char
        tchar = tf.constant(self.t_max, dtype=tf.float32)
        zchar = tf.constant(self.z_max, dtype=tf.float32)
        q0 = tf.abs(self.AI_tf)
        q_scale = tf.maximum(q0, self.kl * Tchar / (zchar + eps))
        stefan_scale = tf.maximum(q0, self.rho * self.Lh * zchar / (tchar + eps))

        # 1. Liquid PDE  (separate, intensity-aware pde_scale)
        Tl_r = self.Tl(self.z_rl, self.t_rl)
        Tl_z = tf.gradients(Tl_r, self.z_rl)[0]
        self.Lr_l = tf.reduce_mean(tf.square(
            (tf.gradients(Tl_r, self.t_rl)[0] - self.alpha_l * tf.gradients(Tl_z, self.z_rl)[0])
            / (self.pde_scale_l + eps)
        ))

        # 2. Solid PDE
        Ts_r = self.Ts(self.z_rs, self.t_rs)
        Ts_z = tf.gradients(Ts_r, self.z_rs)[0]
        self.Lr_s = tf.reduce_mean(tf.square(
            (tf.gradients(Ts_r, self.t_rs)[0] - self.alpha_s * tf.gradients(Ts_z, self.z_rs)[0])
            / (self.pde_scale_s + eps)
        ))

        # 3. IC
        t_zero = tf.zeros_like(self.z0)
        self.LT0 = (
            tf.reduce_mean(tf.square((self.Tl(self.z0, t_zero) - self.T0) / (Tchar + eps))) +
            tf.reduce_mean(tf.square((self.Ts(self.z0, t_zero) - self.T0) / (Tchar + eps)))
        )

        # 4. Surface flux BC: -kl*dTl/dz(0,t) = AI_eff
        z_surf  = tf.zeros_like(self.t_bc)
        Tl_surf = self.Tl(z_surf, self.t_bc)
        self.Lbc_l = tf.reduce_mean(tf.square(
            (-self.kl * tf.gradients(Tl_surf, z_surf)[0] - self.AI_tf) / (q_scale + eps)
        ))

        # 5. Far-field BC
        self.Lbc_s = tf.reduce_mean(tf.square(
            (self.Ts(tf.ones_like(self.t_bc) * self.z_max, self.t_bc) - self.T0) / (Tchar + eps)
        ))

        # 6. Interface temperature
        X_val = self.X(self.t_X)
        self.LXT = tf.reduce_mean(
            tf.square((self.Tl(X_val, self.t_X) - self.Tm) / (Tchar + eps)) +
            tf.square((self.Ts(X_val, self.t_X) - self.Tm) / (Tchar + eps))
        )

        # 7. Stefan condition
        d = tf.maximum(self.delta, tf.constant(1e-9, dtype=tf.float32))
        z_l = tf.maximum(X_val - d, 0.0)
        z_s = tf.minimum(X_val + d, self.z_max)
        s_res = (self.rho * self.Lh * tf.gradients(X_val, self.t_X)[0]
                  - self.ks * tf.gradients(self.Ts(z_s, self.t_X), z_s)[0]
                  + self.kl * tf.gradients(self.Tl(z_l, self.t_X), z_l)[0])
        self.LXS = tf.reduce_mean(tf.square(s_res / (stefan_scale + eps)))

        # 8. X(0)=0
        X0 = self.X(tf.zeros([1, 1], dtype=tf.float32))
        self.LX0 = tf.reduce_mean(tf.square(X0 / (zchar + eps)))

        # 9. Anti-collapse
        self.LXmin = tf.reduce_mean(
            tf.square(tf.nn.relu(self.X_min - X_val) / (self.X_min + eps))
        )

        # 10. Ngwenya: X(t)
        self.L_data_X = tf.reduce_mean(tf.square(
            (self.X(self.t_sup_X) - self.X_sup) / (zchar + eps)
        ))

        # 11. Ngwenya: Ts(z,t)  — normalise by T_char (solid is bounded by Tm)
        self.L_data_Ts = tf.reduce_mean(tf.square(
            (self.Ts(self.z_sup_Ts, self.t_sup_Ts) - self.Ts_sup) / (Tchar + eps)
        ))

        # 12. Ngwenya: Tl(z,t)  — normalise by dT_l (liquid range)
        self.L_data_Tl = tf.reduce_mean(tf.square(
            (self.Tl(self.z_sup_Tl, self.t_sup_Tl) - self.Tl_sup) / (self.dT_l + eps)
        ))

        # physics sub-total (multiplied by curriculum weight)
        physics_loss = (
            self.w_r * (self.Lr_l + self.Lr_s) +
            self.w_T0 * self.LT0 +
            self.w_bc * self.Lbc_l +
            self.w_far * self.Lbc_s +
            self.w_xt * self.LXT +
            self.w_xs * self.LXS +
            self.w_x0 * self.LX0 +
            self.w_xmin * self.LXmin
        )

        # data supervision sub-total (always active)
        data_loss = (
            self.w_data_X * self.L_data_X +
            self.w_data_Ts * self.L_data_Ts +
            self.w_data_Tl * self.L_data_Tl
        )

        self.loss = self.phys_weight * physics_loss + data_loss
        self.data_loss = data_loss
        self.phys_loss = physics_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, data, iters=20000, lr=1e-3, print_every=1000, phys_weight=1.0):
        for it in range(iters + 1):
            feed = {
                self.z_rl: data["z_rl"], self.t_rl: data["t_rl"],
                self.z_rs: data["z_rs"], self.t_rs: data["t_rs"],
                self.z0: data["z0"],
                self.t_bc: data["t_bc"],
                self.t_X: data["t_X"],
                self.t_sup_X: data["t_sup_X"], self.X_sup: data["X_sup"],
                self.z_sup_Ts: data["z_sup_Ts"], self.t_sup_Ts: data["t_sup_Ts"],
                self.Ts_sup: data["Ts_sup"],
                self.z_sup_Tl: data["z_sup_Tl"], self.t_sup_Tl: data["t_sup_Tl"],
                self.Tl_sup: data["Tl_sup"],
                self.lr: lr,
                self.phys_weight: float(phys_weight),
            }
            self.sess.run(self.train_op, feed_dict=feed)

            if it % print_every == 0:
                L, Lp, Ld, Lr_l, Lr_s, LT0, Lbc_l, Lbc_s, LXT, LXS, LX0, LXmin, LdX, LdTs, LdTl = \
                    self.sess.run([
                        self.loss, self.phys_loss, self.data_loss,
                        self.Lr_l, self.Lr_s, self.LT0,
                        self.Lbc_l, self.Lbc_s, self.LXT, self.LXS,
                        self.LX0, self.LXmin,
                        self.L_data_X, self.L_data_Ts, self.L_data_Tl
                    ], feed_dict=feed)
                print(
                    f"it {it:6d} | loss {L:.3e} [p={Lp:.2e} d={Ld:.2e}] | "
                    f"PDE {Lr_l:.2e}/{Lr_s:.2e} | IC {LT0:.2e} | "
                    f"BC {Lbc_l:.2e}/{Lbc_s:.2e} | "
                    f"LXT {LXT:.2e} LXS {LXS:.2e} | "
                    f"Xmin {LXmin:.2e} | "
                    f"[Sup] X={LdX:.2e} Ts={LdTs:.2e} Tl={LdTl:.2e}"
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