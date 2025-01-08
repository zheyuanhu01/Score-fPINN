import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import levy_stable

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--epochs', type=int, default=10001)
parser.add_argument('--N_SM', type=int, default=1000)
parser.add_argument('--N_PINN_LL', type=int, default=1000)
parser.add_argument('--PINN_L', type=int, default=4)
parser.add_argument('--PINN_h', type=int, default=128)
parser.add_argument('--T', type=float, default=0.3)
parser.add_argument('--N', type=int, default=100, help="SDE discretization steps")
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--N_test', type=int, default=10000)
parser.add_argument('--MC_round', type=int, default=int(1e4))
parser.add_argument('--MC_batch', type=int, default=int(1e3))
parser.add_argument('--alpha', type=float, default=1.95)
parser.add_argument('--problem', type=int, default=2)
args = parser.parse_args()
print(args)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

Gamma = np.ones((args.dim, ))

def ND_stable_distribution(alpha, N, dim):
    if alpha == 2: return np.random.randn(N, dim) * np.sqrt(2)
    const = 2 * (np.cos(np.pi * alpha / 4)) ** (2 / alpha)
    rv_A = levy_stable(alpha=alpha/2, beta=1)
    A = rv_A.rvs(size=N) * const
    Z = np.random.randn(N, dim)
    X = Z * np.sqrt(A.reshape(-1, 1))
    return X

def sample_test_points():
    def oneD_stable_distribution(N):
        alpha = args.alpha
        const = 2 * (np.cos(np.pi * alpha / 4)) ** (2 / alpha)
        rv_A = levy_stable(alpha=alpha/2, beta=1)
        A = rv_A.rvs(size=N) * const
        return A
    
    if args.problem == 1: mu = lambda X_t: - X_t * np.tanh(np.linalg.norm(X_t, axis=1, keepdims=True)) # drift
    if args.problem == 2: mu = lambda X_t: - X_t * np.tanh(np.linalg.norm(X_t, axis=1, keepdims=True) / np.sqrt(args.dim)) # drift
    if args.problem == 0: mu = lambda x: -x / args.alpha
    T, N = args.T, args.N

    dt = T / N  # time step size
    
    X = np.random.normal(0, 1, size=(args.N_test, args.dim)) * np.sqrt(Gamma).reshape(1, args.dim)

    # Euler-Maruyama method
    for _ in tqdm(range(1, N + 1)):

        dt = T / N  # time step size
        # dW = np.random.normal(0, np.sqrt(dt), size=(num_simulations, self.dim))  # Wiener process increment

        A = oneD_stable_distribution(args.N_test)
        Z = np.random.normal(size=(args.N_test, args.dim))
        dL = Z * jnp.sqrt(A.reshape(-1, 1))

        dL = dL * (dt ** (1 / args.alpha))

        X = X + mu(X) * dt + dL

    return X, np.ones((args.N_test, )) * T

x, t = sample_test_points()
print("Test data shape of x, t: ", x.shape, t.shape)

from scipy.stats import gaussian_kde
from scipy.special import loggamma
r = np.linalg.norm(x, axis=1)
kde = gaussian_kde(r)
estimated_ll = np.log(kde(r))
dim = args.dim
estimated_ll /= dim
const = (dim - 1) / dim * np.log(r) - (dim - 2) / 2 / dim * np.log(2) - loggamma(dim / 2) / dim + 1 / 2 * np.log(2 * np.pi)
estimated_ll -= const

X, T, Q = x, t, estimated_ll
S = - X / (Gamma.reshape(1, -1) * jnp.exp(-T).reshape(-1, 1) + 2 - 2 * jnp.exp(-T).reshape(-1, 1))
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
        # return - 1 / 2 * jnp.log(2 * np.pi) - 1 / 2 * jnp.mean(jnp.log(Gamma * jnp.exp(-t) + 2 - 2 * jnp.exp(-t))) - jnp.mean(x**2 / (Gamma * jnp.exp(-t) + 2 - 2 * jnp.exp(-t))) / 2
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
        self.T_end = args.T; self.N_discretization = args.N
        self.alpha, self.beta = args.alpha, args.beta
        self.X, self.T, self.Q, self.S = X, T, Q, S

        layers = [args.PINN_h] * (args.PINN_L - 1) + [self.dim]
        @hk.transform
        def network_S1(x, t): return MLP_S1(layers=layers)(x, t)
        self.net_S1 = hk.without_apply_rng(network_S1)
        self.pred_fn_S1 = jax.vmap(self.net_S1.apply, (None, 0, 0))
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
    
    def simulate_sde(self, num_simulations):
        T, N = self.T_end, self.N_discretization
        if args.problem == 1: mu = lambda X_t: - X_t * np.tanh(np.linalg.norm(X_t, axis=1, keepdims=True)) # drift
        if args.problem == 0: mu = lambda x: -x / self.alpha
        if args.problem == 2: mu = lambda X_t: - X_t * np.tanh(np.linalg.norm(X_t, axis=1, keepdims=True) / np.sqrt(args.dim)) # drift

        dt = T / N  # time step size
        t = np.linspace(0, T, N + 1)  # time grid

        # Initial conditions from a Gaussian distribution
        X0 = np.random.normal(0, 1, size=(num_simulations, self.dim)) * np.sqrt(Gamma).reshape(1, self.dim)

        X = np.zeros((N + 1, num_simulations, self.dim))
        X[0] = X0

        # Euler-Maruyama method
        for i in (range(1, N + 1)):
            # dW = np.random.normal(0, np.sqrt(dt), size=(num_simulations, self.dim))  # Wiener process increment

            A = self.oneD_stable_distribution(num_simulations)
            Z = np.random.normal(size=(num_simulations, self.dim))
            dL = Z * jnp.sqrt(A.reshape(-1, 1))

            dL = dL * (dt ** (1 / self.alpha))

            X[i] = X[i-1] + mu(X[i-1]) * dt + dL

        t = np.tile(t.reshape(N + 1, 1), (1, num_simulations))
        # print(t.shape, X.shape)
        t, X = t.reshape(-1), X.reshape(-1, self.dim)

        return t, X

    def smooth_l1_loss(self, pred):
        pred = jnp.abs(pred)
        pred = (pred < self.beta) * pred ** 2 + (pred >= self.beta) * (2 * self.beta * pred - self.beta ** 2)
        return jnp.mean(pred)

    def score_fpinn(self, params, x, t):
        if args.problem == 1: mu = lambda x: -x * jnp.tanh(jnp.linalg.norm(x))
        if args.problem == 0: mu = lambda x: -x / self.alpha
        if args.problem == 2: mu = lambda x: -x * jnp.tanh(jnp.linalg.norm(x) / np.sqrt(self.dim))
        A = lambda x: mu(x) - self.net_S1.apply(params, x, t)
        S2 = lambda x: self.net_S2.apply(self.params_S2, x, t)
        fn = lambda x: - jnp.sum(A(x) * S2(x)) - jnp.sum(jnp.diag(jax.jacrev(A)(x)))
        residual = jax.jacrev(self.net_S2.apply, argnums=2)(self.params_S2, x, t) - jax.jacrev(fn)(x)
        return residual
    def get_loss_score_matching(self, params, x, t):
        pred = jax.vmap(self.score_fpinn, in_axes=(None, 0, 0))(params, x, t)
        loss = self.smooth_l1_loss(pred)
        return loss
    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, rng, xf, tf):
        current_loss, gradients = jax.value_and_grad(self.get_loss_score_matching)(params, xf, tf)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_score1(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_S1)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch)):
            tf, xf = self.simulate_sde(self.N_SM // self.N_discretization)
            current_loss, self.params_S1, self.opt_state, self.rng = self.step(self.params_S1, self.opt_state, self.rng, xf, tf)
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
    def get_loss_score_matching2(self, params, x, t):
        pred = jax.vmap(self.score_matching2, in_axes=(None, 0, 0))(params, x, t)
        loss = jnp.mean(pred)
        return loss
    @partial(jax.jit, static_argnums=(0,))
    def step2(self, params, opt_state, rng, xf, tf):
        current_loss, gradients = jax.value_and_grad(self.get_loss_score_matching2)(params, xf, tf)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_score2(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_S2)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch)):
            tf, xf = self.simulate_sde(self.N_SM // self.N_discretization)
            current_loss, self.params_S2, self.opt_state, self.rng = self.step2(self.params_S2, self.opt_state, self.rng, xf, tf)
            if n % 1000 == 0: print('epoch %d, loss: %e, Score L1: %e'%(n, current_loss, self.L2_pinn_score2(self.params_S2, self.X, self.T, self.S)))
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn_score2(self, params, x, t, s):
        pred = self.pred_fn_S2(params, x, t)
        s, pred = s.reshape(-1), pred.reshape(-1)
        L2_error = jnp.linalg.norm(s - pred, 2) / jnp.linalg.norm(s, 2)
        return L2_error
    
    def residual_pred_ll(self, params, x, t):
        q_t = jax.jacrev(self.net_Q.apply, argnums=2)(params, x, t)
        q_x = jax.jacrev(self.net_Q.apply, argnums=1)(params, x, t)
        if args.problem == 1: mu = lambda x: -x * jnp.tanh(jnp.linalg.norm(x))
        if args.problem == 0: mu = lambda x: -x / self.alpha
        if args.problem == 2: mu = lambda x: -x * jnp.tanh(jnp.linalg.norm(x) / np.sqrt(self.dim))
        fn = lambda x: self.net_S1.apply(self.params_S1, x, t) - mu(x)
        pred_x = jnp.dot(fn(x), q_x) + jnp.mean(jnp.diag(jax.jacrev(fn)(x)))
        return self.dim * (q_t - pred_x)
    def get_loss_pinn_ll(self, params, x, t):
        pred = jax.vmap(self.residual_pred_ll, in_axes=(None, 0, 0))(params, x, t)
        loss = self.smooth_l1_loss(pred)
        return loss
    @partial(jax.jit, static_argnums=(0,))
    def step_Q(self, params, opt_state, rng, xf, tf):
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn_ll)(params, xf, tf)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_ll(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_Q)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch)):
            tf, xf = self.simulate_sde(self.N_SM // self.N_discretization)
            current_loss, self.params_Q, self.opt_state, self.rng = self.step_Q(self.params_Q, self.opt_state, self.rng, xf, tf)
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