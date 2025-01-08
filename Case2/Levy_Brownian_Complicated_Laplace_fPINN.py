import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import levy_stable
import sys

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
parser.add_argument('--N_test', type=int, default=100)
parser.add_argument('--MC_round', type=int, default=int(1e4))
parser.add_argument('--MC_batch', type=int, default=int(1e3))
parser.add_argument('--alpha', type=float, default=1.95)
args = parser.parse_args()
print(args)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

keys = jax.random.split(key, 3)
Gamma = jax.random.uniform(keys[0], shape=(args.dim // 2, )) * 1 + 1
Gamma = jnp.concatenate([Gamma, 1 / Gamma])
A = jax.random.normal(keys[1], shape=(args.dim, args.dim))
A, _ = jnp.linalg.qr(A)
key = keys[2]

A = A.T @ jnp.diag(Gamma)

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
    x = np.random.randn(batch_size, d)
    t = np.random.rand(batch_size, ) * args.T + 1e-2

    def generate(x, t):
        Sigma = (t**3 / 3) * jnp.eye(d)
        Sigma += t * A @ A.T
        Sigma += t ** 2 / 2 * (A + A.T)
        eigenvals, eigenvecs = jnp.linalg.eigh(Sigma)
        Sigma_sqrt = eigenvecs * jnp.sqrt(eigenvals).reshape(1, d) @ eigenvecs.transpose(1, 0) 
        return Sigma_sqrt @ x
    
    x = jax.vmap(generate, in_axes=(0, 0))(x, t)

    x = x + np.random.laplace(size=(batch_size, d))

    x = ND_stable_distribution(alpha, batch_size, d) * t.reshape(-1, 1) ** (1 / args.alpha) + x
    return x, t

x, t = sample_test_points()
print("Test data shape of x, t: ", x.shape, t.shape)

def sample_test(x, t, MCMC_size):
    d, alpha = args.dim, args.alpha
    
    def integrand(x, t):
        # Sigma = (t**3 / 3) * jnp.eye(args.dim)
        # Sigma += t * A @ A.T
        # Sigma += t ** 2 / 2 * (A + A.T)
        # eigvals, eigenvecs = jnp.linalg.eigh(Sigma)
        # logdet = jnp.mean(jnp.log(eigvals))
        # Sigma_inv = (eigenvecs * (1 / eigvals).reshape(1, args.dim)) @ eigenvecs.T
        # return - 1 / 2 * jnp.log(2 * np.pi) - 1 / 2 * logdet - x @ Sigma_inv @ x / 2 / args.dim
        # return - 1 / 2 * jnp.log(2 * np.pi) - 1 / 2 * jnp.mean(jnp.log(sigma)) - jnp.mean(x**2 / sigma) / 2
        return - jnp.log(2) - jnp.mean(jnp.abs(x))

    def func(x, t, y, y0):
        y *= t ** (1 / args.alpha)

        Sigma = (t**3 / 3) * jnp.eye(args.dim)
        Sigma += t * A @ A.T
        Sigma += t ** 2 / 2 * (A + A.T)
        eigvals, eigenvecs = jnp.linalg.eigh(Sigma)
        Sigma_sqrt = eigenvecs * jnp.sqrt(eigvals).reshape(1, d) @ eigenvecs.T

        y0 = Sigma_sqrt @ y0

        y = y + y0

        return integrand(x - y, t)
    
    y = ND_stable_distribution(alpha, MCMC_size, d)
    y0 = np.random.normal(size=(MCMC_size, d))
    logpdf = jax.vmap(jax.vmap(func, in_axes=(None, None, 0, 0)), (0, 0, None, None))(x, t, y, y0) # batch_size, MCMC_size
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

idx = np.argsort(Q)
idx = idx[int(len(idx) * 0.1):]
X, T, Q = X[idx], T[idx], Q[idx]
print(X.shape, T.shape, Q.shape, Gamma.shape)
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
        return X

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
        return X[0]

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
        self.X, self.T, self.Q = X, T, Q
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

        Sigma = jax.vmap(self.Sigma_Test)(self.T) # N_test, dim, dim
        eigenvals, eigenvecs = jnp.linalg.eigh(Sigma)
        Sigma_inv = (eigenvecs * (1 / eigenvals).reshape(-1, 1, self.dim)) @ eigenvecs.transpose(0, 2, 1)
        #print(Sigma_inv.shape, self.X.shape)
        self.S = -jnp.squeeze(Sigma_inv @ self.X.reshape(-1, args.dim, 1))
        #print(self.S.shape)
        self.precompute(self.N_SM, jax.random.PRNGKey(args.SEED + 1))

    def oneD_stable_distribution(self, N):
        alpha = self.alpha
        const = 2 * (np.cos(np.pi * alpha / 4)) ** (2 / alpha)
        rv_A = levy_stable(alpha=alpha/2, beta=1)
        A = rv_A.rvs(size=N) * const
        return A

    def Sigma_Test(self, t):
        Sigma = (1 + t**3 / 3 + t) * jnp.eye(self.dim)
        Sigma += t * A @ A.T
        Sigma += t ** 2 / 2 * (A + A.T)
        return Sigma
    
    def Sigma(self, t):
        Sigma = (t**3 / 3) * jnp.eye(self.dim)
        Sigma += t * A @ A.T
        Sigma += t ** 2 / 2 * (A + A.T)
        return Sigma
    
    def precompute(self, N, rng):
        keys = jax.random.split(rng, 2)
        t = jax.random.uniform(keys[0], shape=(N, )) + 1e-2
        Sigma = jax.vmap(self.Sigma)(t) # N_test, dim, dim
        eigenvals, eigenvecs = jnp.linalg.eigh(Sigma)
        Sigma_sqrt = eigenvecs * jnp.sqrt(eigenvals).reshape(-1, 1, self.dim) @ eigenvecs.transpose(0, 2, 1)
        self.Sigma_mat = Sigma
        self.Sigma_sqrt = Sigma_sqrt
        self.train_t = t
        return keys[1]

    def resample(self, A, N, rng):
        keys = jax.random.split(rng, 4)

        t = self.train_t 
        Sigma = self.Sigma_mat 
        Sigma_sqrt = self.Sigma_sqrt

        x0 = Sigma_sqrt @ jax.random.normal(keys[0], shape=(N, self.dim, 1))
        x0 = jnp.squeeze(x0)
        x0 += jax.random.laplace(keys[1], shape=(N, self.dim))

        Z = jax.random.normal(keys[2], shape=(N, self.dim))
        xf = Z * jnp.sqrt(A.reshape(-1, 1))
        xf = xf * ((t.reshape(-1, 1)) ** (1 / self.alpha)) + x0

        return x0, xf, t, keys[3]

    def resample_LL(self, A, N, rng):
        keys = jax.random.split(rng, 4)

        t = self.train_t 
        Sigma_sqrt = self.Sigma_sqrt

        x0 = Sigma_sqrt @ jax.random.normal(keys[0], shape=(N, self.dim, 1))
        x0 = jnp.squeeze(x0)

        Z = jax.random.normal(keys[1], shape=(N, self.dim))
        xf = Z * jnp.sqrt(A.reshape(-1, 1))
        xf = xf * ((t.reshape(-1, 1)) ** (1 / self.alpha)) + x0

        x0 = jax.random.laplace(keys[2], shape=(N, self.dim))
        xf = x0 + xf
        t0 = jnp.zeros((N, ))

        return x0, xf, t, x0, t0, keys[3]

    def score_matching(self, params, x0, x, t):
        s = self.net_S1.apply(params, x, t) # score prediction
        residual = s * t ** (1 - 1 / self.alpha) + (x - x0) / (self.alpha * t ** (1 / self.alpha))
        return residual
    def score_fpinn(self, params, x, t):
        A_alpha = lambda x: - self.net_S1.apply(params, x, t)
        S2 = lambda x: self.net_S2.apply(self.params_S2, x, t)
        G_S2 = lambda x: (A + t * jnp.eye(self.dim)) @ (A + t * jnp.eye(self.dim)).T @ \
            self.net_S2.apply(self.params_S2, x, t)
        fn = lambda x: 0.5 * jnp.sum(jnp.diag(jax.jacrev(G_S2)(x))) + \
            0.5 * jnp.sum(((A + t * jnp.eye(self.dim)).T @ S2(x)) ** 2) \
            - jnp.sum(A_alpha(x) * S2(x)) - jnp.sum(jnp.diag(jax.jacrev(S2)(x)))
        residual = jax.jacrev(self.net_S2.apply, argnums=2)(self.params_S2, x, t) - jax.jacrev(fn)(x)
        return residual
    def smooth_l1_loss(self, pred):
        pred = jnp.abs(pred)
        pred = (pred < self.beta) * pred ** 2 + (pred >= self.beta) * (2 * self.beta * pred - self.beta ** 2)
        return jnp.mean(pred)
    def get_loss_score_matching(self, params, x0, x, t):
        pred = jax.vmap(self.score_fpinn, in_axes=(None, 0, 0))(params, x, t)
        #pred = jax.vmap(self.score_matching, in_axes=(None, 0, 0, 0))(params, x0, x, t)
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
    

    def score_matching2(self, params, x, t):
        s = 0.5 * jnp.sum(self.net_S2.apply(params, x, t)**2)
        fn = lambda x: self.net_S2.apply(params, x, t)
        nabla_s = jnp.sum(jnp.diag(jax.jacfwd(fn)(x)))
        return nabla_s + s
    def boundary_pred_s2(self, params, x, t):
        s = self.net_S2.apply(params, x, t)
        s_ref = - jnp.sign(x)
        return (s - s_ref)
    def get_loss_score_matching2(self, params, x, t, xb, tb):
        pred = jax.vmap(self.score_matching2, in_axes=(None, 0, 0))(params, x, t)
        loss = jnp.mean(pred)
        # pred_b = jax.vmap(self.boundary_pred_s2, in_axes=(None, 0, 0))(params, xb, tb)
        # loss_b = jnp.mean(pred_b ** 2)
        return loss #+ 1 * loss_b
    @partial(jax.jit, static_argnums=(0,))
    def step2(self, params, opt_state, rng, A):
        x0, xf, tf, xb, tb, rng = self.resample_LL(A, self.N_PINN_LL, rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_score_matching2)(params, xf, tf, xb, tb)
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
        fn = lambda x: (A + t * jnp.eye(self.dim)) @ (A + t * jnp.eye(self.dim)).T @ \
            jax.jacrev(self.net_Q.apply, argnums=1)(params, x, t)
        pred_x = 0.5 * jnp.mean(((A + t * jnp.eye(self.dim)).T @ s)**2) + \
            0.5 * jnp.sum(jnp.diag(jax.jacrev(fn)(x)))
        pred_x += jnp.mean(S_alpha * s) + nabla_dot_S_alpha
        return self.dim * (q_t - pred_x)
    
    def boundary_pred_ll(self, params, x, t):
        q = self.net_Q.apply(params, x, t)
        q_ref = - jnp.log(2) - jnp.mean(jnp.abs(x))
        return self.dim * (q - q_ref)
    
    def get_loss_pinn_ll(self, params, x0, x, t, xb, tb):
        pred = jax.vmap(self.residual_pred_ll, in_axes=(None, 0, 0))(params, x, t)
        loss = self.smooth_l1_loss(pred)
        pred_b = jax.vmap(self.boundary_pred_ll, in_axes=(None, 0, 0))(params, xb, tb)
        return 1 * loss + 100 * jnp.mean(pred_b**2)
    @partial(jax.jit, static_argnums=(0,))
    def step_Q(self, params, opt_state, rng, A):
        x0, xf, tf, xb, tb, rng = self.resample_LL(A, self.N_PINN_LL, rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn_ll)(params, x0, xf, tf, xb, tb)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_ll(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_Q)
        self.rng = jax.random.PRNGKey(self.SEED)
        current_loss = 0
        for n in tqdm(range(self.epoch)):
            if n % 1000 == 0: print('epoch %d, loss %e, LL L2/Linf %e %e, PDF L2/Linf %e/%e'%(n, current_loss, *self.L2_pinn_Q(self.params_Q, self.X, self.T, self.Q)))
            A = self.oneD_stable_distribution(self.N_SM)
            current_loss, self.params_Q, self.opt_state, self.rng = self.step_Q(self.params_Q, self.opt_state, self.rng, A)
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