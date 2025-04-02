from functools import partial
import random
import jax
import jax.numpy as jnp
import optax

from src.utils import decode_fsm, entropy, prepare_str, get_separate_char, decode_str
from src.transducers.transducers import TensorTransducer, FunctionTransducer, FSM, Params, Stats, TrainState, TrainResult



def loss_f(params, x, y0, entropy_weight, hard=False):
  T, R, s0 = decode_fsm(params, hard=hard)
  fsm = FSM(T, R, s0)

  ### Se puede crear un TensorTransducer y sacar la funcion 'run_fsm_with_values'
  y, s = TensorTransducer.run_fsm_with_values(x, fsm.R, fsm.T, fsm.s0)
  error = jnp.square(y-y0).sum()
  entropy_loss = entropy(s.mean(0)) * entropy_weight
  total = error + entropy_loss
  states_used = s.max(0).sum()
  return total, Stats(total=total, error=error, entropy=entropy_loss, states_used=states_used)


class Learner:
  def __init__(self, max_states, alphabet_in, alphabet_out, entropy_weight=0, lazy_bias=1.0, train_step_n=1000, run_n=1000, epsilon_error= 0.000001, learning_rate=0.25, b1=0.5, b2=0.5, verbose=0):
    self.target_transducer = None
    self.separate_char = None
    self.max_length_sec = 10 ### Cambiar
    self.xs = []
    self.ys = []
    self.r = None

    self.max_states = max_states
    self.CHAR_IN = len(alphabet_in)
    self.CHAR_OUT = len(alphabet_out)
    self.alphabet_in = alphabet_in
    self.alphabet_out = alphabet_out
    self.separate_char = get_separate_char(self.alphabet_in + self.alphabet_out)
    self.alphabet_in_ext = alphabet_in + [self.separate_char]
    self.alphabet_out_ext = alphabet_out + [self.separate_char]
    self.entropy_weight = entropy_weight
    self.lazy_bias = lazy_bias
    self.train_step_n = train_step_n
    self.run_n = run_n
    self.epsilon_error = epsilon_error
    self.optimizer = optax.adam(learning_rate, b1, b2)
    self.verbose = verbose
    self.loss_f = None


  @partial(jax.jit, static_argnums=(0,))
  def train_step(self, train_state):
    params, opt_state = train_state
    grad_f = jax.grad(self.loss_f, has_aux=True)
    grads, stats = grad_f(params)
    updates, opt_state = self.optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return TrainState(params, opt_state), stats

  def run(self, key):
    logs = []
    params0 = self.init_fsm(key)
    opt_state = self.optimizer.init(params0)
    train_state = TrainState(params0, opt_state)

    for _ in range(self.train_step_n):
      train_state, stats = self.train_step(train_state)
      logs.append(stats)

    _, evaluation = self.loss_f(train_state.params, hard=True)
    return TrainResult(train_state.params, evaluation, logs)

  def init_fsm(self, key, noise=1e-3):
    k1, k2, k3 = jax.random.split(key, 3)
    T = jax.random.normal(k1, [self.CHAR_IN+1, self.max_states, self.max_states]) * noise
    T += jnp.eye(self.max_states) * self.lazy_bias
    R = jax.random.normal(k2, [self.CHAR_IN+1, self.max_states, self.CHAR_OUT+1]) * noise
    s0 = jax.random.normal(k3, [self.max_states]) * noise
    return Params(T, R, s0)

  def contain_query(self, x):
    return self.target_transducer(x)

  def equivalence_query(self, transducer, t=10):
    to_test = []
    for _ in range(t):
      test = self.generate_input(random.randint(1, self.max_length_sec))
      to_test.append(test)
      if self.target_transducer.run_fsm(test) != transducer.run_fsm(test):
        return False, test

    return True, None

  def generate_input(self, n):
    return "".join([random.choice(self.target_transducer.alphabet_in) for _ in range(n)] + [self.separate_char])

  def generate_keys(self, run_n):
    key = jax.random.PRNGKey(1)
    return jax.random.split(key, run_n)

  def train_fsm(self, keys, x, y):
    x, y = prepare_str(x, self.alphabet_in_ext), prepare_str(y, self.alphabet_out_ext)
    self.loss_f = partial(loss_f, x=x, y0=y, entropy_weight=self.entropy_weight)
    self.r = jax.vmap(self.run)(keys)
    best_i = (self.r.eval.states_used + self.r.eval.error*10000).argmin()
    best_params = jax.tree_util.tree_map(lambda a:a[best_i], self.r.params)
    T, R, s0 = decode_fsm(best_params, hard=True)
    best_fsm = FSM(T, R, s0)
    return best_fsm.T, best_fsm.R, best_fsm.s0

  def generate_xy(self):
    return "".join(self.xs), "".join(self.ys)

  def learn(self, target_transducer, budget, run_n=1000, verbose=0):
    ### Assert alphabets in target_transducer are the same in self

    self.target_transducer = target_transducer
    self.xs = [self.generate_input(random.randint(1, self.max_length_sec))]
    self.ys = [self.target_transducer.run_fsm(self.xs[0])]

    transducer = None
    for i in range(budget):
      keys = self.generate_keys(run_n)
      x_test, y_test = self.generate_xy()
      T, R, s0 = self.train_fsm(keys, x_test, y_test)
      transducer = TensorTransducer(T, R, s0, self.alphabet_in, self.alphabet_out, self.max_states)

      if verbose:
        print(f"Iteration: {i}")
        print(f"xs: {x_test}")
        print(f"ys: {y_test}")
        print(f"y_predict: {transducer.run_fsm(x_test)}")
        print(f"Error: {self.r.eval.error.min()}")

      res, counter = self.equivalence_query(transducer)
      if res:
        break

      self.xs.append(counter)
      self.ys.append(self.target_transducer.run_fsm(counter))

    return transducer

  def learn_from_dataset(self, xs, ys, verbose=0):
    assert len(xs) == len(ys), "Error"
    keys = self.generate_keys(self.run_n)

    x = "".join([x + self.separate_char for x in xs])
    y = "".join([y + self.separate_char for y in ys])
    T, R, s0 = self.train_fsm(keys, x, y)

    return TensorTransducer(T, R, s0, self.alphabet_in, self.alphabet_out, self.max_states)




