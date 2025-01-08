import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import multivariate_t
from scipy.special import gamma
from scipy.stats import levy_stable

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--epochs', type=int, default=100001)
parser.add_argument('--N_SM', type=int, default=10000)
parser.add_argument('--N_PINN_LL', type=int, default=10000)
parser.add_argument('--PINN_L', type=int, default=4)
parser.add_argument('--PINN_h', type=int, default=128)
parser.add_argument('--T', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--N_test', type=int, default=10000)
parser.add_argument('--MC_round', type=int, default=int(1e4))
parser.add_argument('--MC_batch', type=int, default=int(1e3))
parser.add_argument('--alpha', type=float, default=1.95)
args = parser.parse_args()
print(args)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

def gen_gaussian_cov():
    Gamma = np.random.rand(args.dim // 2, ) * 1 + 1
    Gamma = np.concatenate([Gamma, 1 / Gamma])
    return Gamma

Gamma = gen_gaussian_cov()

def ND_stable_distribution(alpha, N, dim):
    if alpha == 2: return np.random.randn(N, dim) * np.sqrt(2)
    const = 2 * (np.cos(np.pi * alpha / 4)) ** (2 / alpha)
    rv_A = levy_stable(alpha=alpha/2, beta=1)
    A = rv_A.rvs(size=N) * const
    Z = np.random.randn(N, dim)
    X = Z * np.sqrt(A.reshape(-1, 1))
    return X

def sample_test_points():
    d, alpha = args.dim, args.alpha
    batch_size = args.N_test
    x0 = np.random.randn(batch_size, d) * np.sqrt(Gamma).reshape(1, d)
    t = np.random.rand(batch_size, ) * args.T + 1e-2
    x = ND_stable_distribution(alpha, batch_size, d) * t.reshape(-1, 1) ** (1 / args.alpha) + \
        x0 + np.random.randn(batch_size, d) * np.sqrt(t).reshape(-1, 1)
    return x, t

x, t = sample_test_points()
print("Test data shape of x, t: ", x.shape, t.shape)

def sample_test(x, t, MCMC_size):
    d, alpha = args.dim, args.alpha
    
    def integrand(x, t):
        sigma = Gamma + t
        return - 1 / 2 * jnp.log(2 * np.pi) - 1 / 2 * jnp.mean(jnp.log(sigma)) - jnp.mean(x**2 / sigma) / 2

    def func(x, t, y):
        y *= t ** (1 / args.alpha)
        return integrand(x - y, t)
    
    y = ND_stable_distribution(alpha, MCMC_size, d)
    logpdf = jax.vmap(jax.vmap(func, in_axes=(None, None, 0)), (0, 0, None))(x, t, y) # batch_size, MCMC_size
    logpdf *= d
    return logpdf

Q = []
for _ in tqdm(range(args.MC_round)):
    q = sample_test(x, t, args.MC_batch)
    q_max = q.max(1)
    q = q - q_max.reshape(-1, 1)
    q = jnp.exp(q)
    q = jnp.mean(q, 1)
    q = jnp.log(q) + q_max
    Q.append(q)

q = jnp.stack(Q, 1)
q_max = q.max(1)
q = q - q_max.reshape(-1, 1)
q = jnp.exp(q)
q = jnp.mean(q, 1)
q = jnp.log(q) + q_max
q /= args.dim
print(q.min(), q.max())

X, T, Q = x, t, q
S = - X / (Gamma.reshape(1, -1) + 2 * T.reshape(-1, 1))
print(X.shape, T.shape, Q.shape, Gamma.shape, S.shape)
print(Q.min(), Q.max(), Q.min())

idx = np.argsort(Q)
idx = idx[int(len(idx) * 0.1):]
X, T, Q, S = X[idx], T[idx], Q[idx], S[idx]
print(X.shape, T.shape, Q.shape, Gamma.shape, S.shape)
print(Q.min(), Q.mean(), Q.max())

class MLP_S1(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x, t):
        X = jnp.hstack([x, t])
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        return X

class MLP_S2(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x, t):
        X = jnp.hstack([x, t])
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        return X * t - x / Gamma
    
class MLP_Q(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x, t):
        X = jnp.hstack([x, t])
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        X = X[0] * t - 1 / 2 * jnp.log(2 * np.pi) - 1 / 2 * jnp.mean(jnp.log(Gamma)) - jnp.mean(x**2 / (Gamma)) / 2
        return X

class PINN:
    def __init__(self, args):
        self.dim = args.dim; self.epoch = args.epochs; self.SEED = args.SEED;
        self.N_SM, self.N_PINN_LL = args.N_SM, args.N_PINN_LL
        self.T_end = args.T
        self.alpha, self.beta = args.alpha, args.beta
        layers = [args.PINN_h] * (args.PINN_L - 1) + [self.dim]
        @hk.transform
        def network_S1(x, t): return MLP_S1(layers=layers)(x, t)
        self.net_S1 = hk.without_apply_rng(network_S1)
        self.pred_fn_S1 = jax.vmap(self.net_S1.apply, (None, 0, 0))
        self.X, self.T, self.Q, self.S = X, T, Q, S
        self.params_S1 = self.net_S1.init(key, self.X[0], self.T[0])

        layers = [args.PINN_h] * (args.PINN_L - 1) + [self.dim]
        @hk.transform
        def network_S2(x, t): return MLP_S2(layers=layers)(x, t)
        self.net_S2 = hk.without_apply_rng(network_S2)
        self.pred_fn_S2 = jax.vmap(self.net_S2.apply, (None, 0, 0))
        self.params_S2 = self.net_S2.init(key, self.X[0], self.T[0])

        layers_Q = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network_Q(x, t): return MLP_Q(layers=layers_Q)(x, t)
        self.net_Q = hk.without_apply_rng(network_Q)
        self.pred_fn_Q = jax.vmap(self.net_Q.apply, (None, 0, 0))
        self.params_Q = self.net_Q.init(key, self.X[0], self.T[0])

    def oneD_stable_distribution(self, N):
        alpha = self.alpha
        const = 2 * (np.cos(np.pi * alpha / 4)) ** (2 / alpha)
        rv_A = levy_stable(alpha=alpha/2, beta=1)
        A = rv_A.rvs(size=N) * const
        return A

    def resample(self, A, N, rng):
        keys = jax.random.split(rng, 5)
        tf = jax.random.uniform(keys[1], shape=(N,)) * self.T_end + 1e-2
        x0 = jax.random.normal(keys[0], shape=(N, self.dim)) * \
            jnp.sqrt(Gamma.reshape(1, self.dim) + tf.reshape(N, 1))

        Z = jax.random.normal(keys[2], shape=(N, self.dim))
        xf = Z * jnp.sqrt(A.reshape(-1, 1))

        xf = xf * ((tf.reshape(-1, 1)) ** (1 / self.alpha)) + x0
        return x0, xf, tf, keys[4]

    def smooth_l1_loss(self, pred):
        pred = jnp.abs(pred)
        pred = (pred < self.beta) * pred ** 2 + (pred >= self.beta) * (2 * self.beta * pred - self.beta ** 2)
        return jnp.mean(pred)

    def score_fpinn(self, params, x, t):
        A = lambda x: - self.net_S1.apply(params, x, t)
        S2 = lambda x: self.net_S2.apply(self.params_S2, x, t)
        fn = lambda x: 0.5 * jnp.sum(jnp.diag(jax.jacrev(S2)(x))) + 0.5 * jnp.sum(S2(x) ** 2) \
            - jnp.sum(A(x) * S2(x)) - jnp.sum(jnp.diag(jax.jacrev(S2)(x)))
        residual = jax.jacrev(self.net_S2.apply, argnums=2)(self.params_S2, x, t) - jax.jacrev(fn)(x)
        return residual
    
    def score_matching(self, params, x0, x, t):
        s = self.net_S1.apply(params, x, t) # score prediction
        residual = s * t ** (1 - 1 / self.alpha) + (x - x0) / (self.alpha * t ** (1 / self.alpha))
        return residual
    def get_loss_score_matching(self, params, x0, x, t):
        # pred = jax.vmap(self.score_matching, in_axes=(None, 0, 0, 0))(params, x0, x, t)
        pred = jax.vmap(self.score_fpinn, in_axes=(None, 0, 0))(params, x, t)
        loss = self.smooth_l1_loss(pred)
        return loss
    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, rng, A):
        x0, xf, tf, rng = self.resample(A, self.N_SM, rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_score_matching)(params, x0, xf, tf)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_score1(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_S1)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch)):
            A = self.oneD_stable_distribution(self.N_SM)
            current_loss, self.params_S1, self.opt_state, self.rng = self.step(self.params_S1, self.opt_state, self.rng, A)
            if n % 1000 == 0: print('epoch %d, loss: %e, Score L1: %e'%(n, current_loss, self.L2_pinn_score(self.params_S1, self.X, self.T, self.S)))
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn_score(self, params, x, t, s):
        pred = self.pred_fn_S1(params, x, t)
        s, pred = s.reshape(-1), pred.reshape(-1)
        L2_error = jnp.linalg.norm(s - pred, 2) / jnp.linalg.norm(s, 2)
        return L2_error
    
    def score_matching2(self, params, x0, x, t):
        s = self.net_S2.apply(params, x, t) # score prediction
        # residual = s * jnp.exp(-t * 2 / self.alpha) + x0 / Gamma
        residual = s + x0 / (Gamma + t)
        return residual
    def get_loss_score_matching2(self, params, x0, x, t):
        pred = jax.vmap(self.score_matching2, in_axes=(None, 0, 0, 0))(params, x0, x, t)
        loss = self.smooth_l1_loss(pred)
        return loss
    @partial(jax.jit, static_argnums=(0,))
    def step2(self, params, opt_state, rng, A):
        x0, xf, tf, rng = self.resample(A, self.N_SM, rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_score_matching2)(params, x0, xf, tf)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_score2(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_S2)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch)):
            A = self.oneD_stable_distribution(self.N_SM)
            current_loss, self.params_S2, self.opt_state, self.rng = self.step2(self.params_S2, self.opt_state, self.rng, A)
            if n % 1000 == 0: print('epoch %d, loss: %e, Score L1: %e'%(n, current_loss, self.L2_pinn_score2(self.params_S2, self.X, self.T, self.S)))
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn_score2(self, params, x, t, s):
        pred = self.pred_fn_S2(params, x, t)
        s, pred = s.reshape(-1), pred.reshape(-1)
        L2_error = jnp.linalg.norm(s - pred, 2) / jnp.linalg.norm(s, 2)
        return L2_error
    
    def residual_pred_ll(self, params, x, t):
        q_t = jax.jacrev(self.net_Q.apply, argnums=2)(params, x, t)
        s = jax.jacrev(self.net_Q.apply, argnums=1)(params, x, t) * self.dim
        S_alpha = self.net_S1.apply(self.params_S1, x, t)
        nabla_dot_S_alpha = jnp.mean(jnp.diag(jax.jacrev(self.net_S1.apply, argnums=1)(self.params_S1, x, t)))
        nabla_dot_s = jnp.sum(jnp.diag(jax.hessian(self.net_Q.apply, argnums=1)(params, x, t)))
        pred_x = 0.5 * nabla_dot_s + 0.5 * jnp.mean(s**2) + jnp.mean(S_alpha * s) + nabla_dot_S_alpha
        return self.dim * (q_t - pred_x)
    def get_loss_pinn_ll(self, params, x0, x, t):
        pred = jax.vmap(self.residual_pred_ll, in_axes=(None, 0, 0))(params, x, t)
        loss = self.smooth_l1_loss(pred)
        return loss
    @partial(jax.jit, static_argnums=(0,))
    def step_Q(self, params, opt_state, rng, A):
        x0, xf, tf, rng = self.resample(A, self.N_PINN_LL, rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn_ll)(params, x0, xf, tf)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_ll(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_Q)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch // 10)):
            A = self.oneD_stable_distribution(self.N_SM)
            current_loss, self.params_Q, self.opt_state, self.rng = self.step_Q(self.params_Q, self.opt_state, self.rng, A)
            if n % 1000 == 0: print('epoch %d, loss %e, LL L2/Linf %e %e, PDF L2/Linf %e/%e'%(n, current_loss, *self.L2_pinn_Q(self.params_Q, self.X, self.T, self.Q)))
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn_Q(self, params, x, t, q):
        d = self.dim
        pred = self.pred_fn_Q(params, x, t)
        LL_L2_error = jnp.linalg.norm(q - pred, 2) / jnp.linalg.norm(q, 2)
        LL_Linf_error = jnp.max(jnp.abs(q - pred)) / jnp.max(jnp.abs(q))
        q, pred = q - q.max(), pred - q.max()
        PDF_L2_error = jnp.linalg.norm(jnp.exp(d * q) - jnp.exp(d * pred), 2) / jnp.linalg.norm(jnp.exp(d * q), 2)
        PDF_Linf_error = jnp.max(jnp.abs(jnp.exp(d * q) - jnp.exp(d * pred))) / jnp.max(jnp.exp(d * q))
        return LL_L2_error, LL_Linf_error, PDF_L2_error, PDF_Linf_error

model = PINN(args)
model.train_score2()
model.train_score1()
model.train_ll()