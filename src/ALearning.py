from functools import partial
import random
import jax
import jax.numpy as jnp
import optax

from src.utils import decode_fsm, entropy, prepare_str, FSM, Params, Stats, TrainState, TrainResult, get_separate_char, decode_str
from src.transducers import TensorTransducer, FunctionTransducer



def loss_f(params, x, y0, entropy_weight, hard=False):
  fsm = decode_fsm(params, hard=hard)

  ### Se puede crear un TensorTransducer y sacar la funcion 'run_fsm_with_values'
  y, s = TensorTransducer.run_fsm_with_values(x, fsm.R, fsm.T, fsm.s0)
  print("-----------------")
  print(f"x = {x}")
  print(f"y = {y}")
  print("-----------------")
  error = jnp.square(y-y0).sum()
  entropy_loss = entropy(s.mean(0)) * entropy_weight
  total = error + entropy_loss
  states_used = s.max(0).sum()
  return total, Stats(total=total, error=error, entropy=entropy_loss, states_used=states_used)

class Trainer:
  def __init__(self, x, y0, alphabet_in, alphabet_out, states, entropy_weight=0.01, lazy_bias=1.0, train_step_n=1000):
    self.STATE_MAX = states
    self.CHAR_IN = len(alphabet_in)
    self.CHAR_OUT = len(alphabet_out)
    self.alphabet_in = alphabet_in
    self.alphabet_out = alphabet_out
    x, y0 = prepare_str(x, alphabet_in), prepare_str(y0, alphabet_out)
    self.task = (x, y0)
    self.loss_f = partial(loss_f, x=x, y0=y0, entropy_weight=entropy_weight)
    self.lazy_bias = lazy_bias
    self.train_step_n = train_step_n
    self.epsilon_error = 0.000001
    self.optimizer = optax.adam(0.25, 0.5, 0.5)



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
    params0 = self.init_fsm(key, lazy_bias=self.lazy_bias)
    opt_state = self.optimizer.init(params0)
    train_state = TrainState(params0, opt_state)

    for _ in range(self.train_step_n):
      train_state, stats = self.train_step(train_state)
      logs.append(stats)

    _, eval = self.loss_f(train_state.params, hard=True)
    return TrainResult(train_state.params, eval, logs)

  def init_fsm(self, key, lazy_bias=1.0, noise=1e-3) -> Params:
    k1, k2, k3 = jax.random.split(key, 3)
    T = jax.random.normal(k1, [self.CHAR_IN, self.STATE_MAX, self.STATE_MAX]) * noise
    T += jnp.eye(self.STATE_MAX) * lazy_bias
    R = jax.random.normal(k2, [self.CHAR_IN, self.STATE_MAX, self.CHAR_OUT]) * noise
    s0 = jax.random.normal(k3, [self.STATE_MAX]) * noise
    return Params(T, R, s0)





class Learner:
  def __init__(self):
    self.target_transducer = None
    self.separate_char = None
    self.max_length_sec = 10 ### Cambiar
    self.xs = []
    self.ys = []
    self.r = None

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

  def train_fsm(self, keys, x, y, alphabet_in, alphabet_out, state_max, entropy_weight=0, verbose=0):
    trainer = Trainer(x, y, alphabet_in, alphabet_out, state_max, entropy_weight)
    self.r = jax.vmap(trainer.run)(keys)

    best_i = (self.r.eval.states_used + self.r.eval.error*10000).argmin()
    print(f"Best i = {best_i}")
    print(f"The best: {self.r.eval.error.min()}")
    best_params = jax.tree_util.tree_map(lambda x:x[best_i], self.r.params)
    best_fsm = decode_fsm(best_params, hard=True)


    return best_fsm.T, best_fsm.R, best_fsm.s0

  def generate_xy(self):
    return "".join(self.xs), "".join(self.ys)

  def learn(self, target_transducer, budget, run_n=1000, verbose=0):
    self.target_transducer = target_transducer
    self.separate_char = get_separate_char(target_transducer.alphabet_in + target_transducer.alphabet_out)
    self.xs = [self.generate_input(random.randint(1, self.max_length_sec))]
    self.ys = [self.target_transducer.run_fsm(self.xs[0])]

    transducer = None
    for i in range(budget):
      keys = self.generate_keys(run_n)
      x_test, y_test = self.generate_xy()
      T, R, s0 = self.train_fsm(keys, x_test, y_test, self.target_transducer.alphabet_in_ext, self.target_transducer.alphabet_out_ext, self.target_transducer.STATE_MAX)
      transducer = TensorTransducer(T, R, s0, self.target_transducer.alphabet_in, self.target_transducer.alphabet_out, self.target_transducer.MAX_STATE)  ### Convierte el fsm en Transducer con tensores

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
  
  def learn_from_dataset(self, xs, ys, alphabet_in, alphabet_out, run_n=1000, state_max=8, verbose=0):
    assert len(xs) == len(ys), "Error"

    self.separate_char = get_separate_char(alphabet_in + alphabet_out)
    keys = self.generate_keys(run_n)
    
    x = "".join([x + self.separate_char for x in xs])
    y = "".join([y + self.separate_char for y in ys])

    alphabet_in_ext = alphabet_in + [self.separate_char]
    alphabet_out_ext = alphabet_out + [self.separate_char]
    T, R, s0 = self.train_fsm(keys, x, y, alphabet_in_ext, alphabet_out_ext, state_max=state_max)

    transducer = TensorTransducer(T, R, s0, alphabet_in, alphabet_out, state_max)
    return transducer

    
  

