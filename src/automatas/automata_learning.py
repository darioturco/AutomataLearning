from functools import partial
import random
import jax
import jax.numpy as jnp
import optax

from src.utils import decode_fsm, entropy, prepare_str, get_separate_char, decode_str, probabilistic_sample
from src.automatas.automatas import TensorAutomata, FunctionAutomata, StateAutomata, FSM, Params, Stats, TrainState, TrainResult
from src.automatas.algorithms.k_tails import KTail

def loss_f(params, x, y0, entropy_weight, hard=False):
	T, A, s0 = decode_fsm(params, hard=hard)
	fsm = FSM(T, A, s0)
	y, s = TensorAutomata.run_fsm_with_values(x, fsm.A, fsm.T, fsm.s0)
	y = y.transpose(1,0,2)
	error = jnp.square(y - y0).sum()
	entropy_loss = entropy(s.mean(0)) * entropy_weight
	total = error + entropy_loss
	states_used = s.max(0).sum()
	return total, Stats(total=total, error=error, entropy=entropy_loss, states_used=states_used)


class Learner:
	def __init__(self, max_states, alphabet, entropy_weight=0, lazy_bias=1.0, train_step_n=1000, run_n=1000, learning_rate=0.25, b1=0.5, b2=0.5, verbose=0):
		self.target_automata = None
		self.separate_char = None
		self.xs = []
		self.ys = []
		self.r = None

		self.max_states = max_states
		self.CHAR_IN = len(alphabet)
		self.alphabet = alphabet
		self.separate_char = get_separate_char(self.alphabet + ['0', '1'])
		self.alphabet_ext = alphabet + [self.separate_char]
		self.entropy_weight = entropy_weight
		self.lazy_bias = lazy_bias
		self.train_step_n = train_step_n
		self.run_n = run_n
		self.optimizer = optax.adam(learning_rate, b1, b2)
		self.verbose = verbose
		self.loss_f = None
		self.k_tail_learner = KTail(self)

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

		for i in range(self.train_step_n):
			train_state, stats = self.train_step(train_state)
			logs.append(stats)

		_, evaluation = self.loss_f(train_state.params, hard=True)
		return TrainResult(train_state.params, evaluation, logs)


	def init_fsm(self, key, noise=1e-3):
		k1, k2, k3 = jax.random.split(key, 3)
		T = jax.random.normal(k1, [self.CHAR_IN + 1, self.max_states, self.max_states]) * noise
		T += jnp.eye(self.max_states) * self.lazy_bias
		A = jax.random.normal(k2, [self.max_states, 3]) * noise
		s0 = jax.random.normal(k3, [self.max_states]) * noise
		return Params(T, A, s0)

	@staticmethod
	def contain_query(self, x, target_automata):
		return target_automata(x)

	@staticmethod
	def equivalence_query(self, automata, target_automata, t, p=0.7):
		to_test = []
		for _ in range(t):
			test = probabilistic_sample(target_automata.alphabet, p=p)
			to_test.append(test)
			if target_automata.run_fsm(test) != automata.run_fsm(test):
				return False, test

		return True, None

	def generate_keys(self, run_n):
		key = jax.random.PRNGKey(1)
		return jax.random.split(key, run_n)



	def train_fsm(self, keys, x, y, concatenate=False):
		#alphabet_in = self.alphabet_ext if concatenate else self.alphabet
		#alphabet_out = ['0', '1'] + ([self.separate_char] if concatenate else [])
		alphabet_in = self.alphabet_ext
		alphabet_out = ['0', '1'] + [self.separate_char]
		if concatenate:
			xs = jnp.array([prepare_str(x_, alphabet_in) for x_ in x])	### Va a explotar cuando concatenate no sea True
			ys = jnp.array([prepare_str(y_, alphabet_out) for y_ in y])
		else:
			max_len = max([len(x_) for x_ in x])
			xs = jnp.array([prepare_str(x_, alphabet_in, padding=max_len-len(x_)) for x_ in x])
			ys = jnp.array([prepare_str(y_, alphabet_out, padding=max_len-len(y_)) for y_ in y])

		#xs, ys = , prepare_str(y, alphabet_out)
		self.loss_f = partial(loss_f, x=xs, y0=ys, entropy_weight=self.entropy_weight)
		self.r = jax.vmap(self.run)(keys)
		best_i = (self.r.eval.states_used + self.r.eval.error * 10000).argmin()
		best_params = jax.tree_util.tree_map(lambda a: a[best_i], self.r.params)
		T, A, s0 = decode_fsm(best_params, hard=True)
		best_fsm = FSM(T, A, s0)
		return best_fsm.T, best_fsm.A, best_fsm.s0

	def generate_xy(self):
		return "".join(self.xs), "".join(self.ys)

	def learn(self, target_automata, budget, t, p, run_n=1000, verbose=0):
		assert target_automata.alphabet == self.alphabet, "Error, alphabet should have the same length."

		self.target_automata = target_automata
		self.xs = [probabilistic_sample(self.target_automata.alphabet, t, p)]
		self.ys = ["".join(['1' if self.target_automata.run_fsm(xs[:i+1]) else '0' for i in range(len(xs))]) for xs in self.xs]

		automata = None
		for i in range(budget):
			keys = self.generate_keys(run_n)
			x_test, y_test = self.generate_xy()
			T, R, s0 = self.train_fsm(keys, x_test, y_test)
			automata = TensorAutomata(T, R, s0, self.alphabet, self.max_states)

			if verbose:
				print(f"Iteration: {i}")
				print(f"xs: {x_test}")
				print(f"ys: {y_test}")
				print(f"y_predict: {automata.run_fsm(x_test)}")
				print(f"Error: {self.r.eval.error.min()}")

			res, counter = self.equivalence_query(automata, t, p)
			if res:
				break

			self.xs.append(counter)
			self.ys.append(self.target_automata.run_fsm(counter))

		return automata



	def learn_from_k_tail(self, xs, k, verbose=0):
		return self.k_tail_learner.learn(xs, k, verbose=verbose)

	def learn_from_dataset(self, xs, ys, run_n=1000, concatenate=False, verbose=0):
		assert len(xs) == len(ys), "Error"

		keys = self.generate_keys(run_n)
		if concatenate:
			xs = ["".join([x + self.separate_char for x in xs])]
			ys = ["".join([y + self.separate_char for y in ys])]

		T, A, s0 = self.train_fsm(keys, xs, ys, concatenate)

		return TensorAutomata(T, A, s0, self.alphabet, self.max_states)

	def build_pta(self, xs):
		states = set()
		accepting_states = set()
		edges = []

		for x in xs:
			current_state = ""
			next_state = ""
			for symbol in x:
				next_state = current_state + symbol
				states.add(current_state)
				states.add(next_state)
				edges[current_state] = [(current_state, symbol, next_state)]
				current_state = next_state

			accepting_states.add(next_state)

		return StateAutomata(states, edges, accepting_states, "", self.alphabet)

	def merge_states(self, automata, k):
		state_list = list(automata.states)
		i = 0
		while i < len(state_list):
			state1 = state_list[i]
			j = i + 1
			while j < len(state_list):
				state2 = state_list[j]
				if self.get_k_tails(state1, k) == self.get_k_tails(state2, k):
					# Merge state2 into state1
					for symbol, targets in state2.transitions.items():
						for target in targets:
							state1.add_transition(symbol, target)
					# Remove state2 and continue
					automata.states.pop(state2.name)
					state_list.remove(state2)
				else:
					j += 1
			i += 1

	def get_k_tails(self, state, k):
		"""Helper function to get all tails of length k starting from a given state."""
		tails = set()

		def dfs(current_state, path):
			if len(path) == k:
				tails.add(tuple(path))
				return
			if current_state in state.transitions:
				for symbol, next_states in state.transitions[current_state].items():
					for next_state in next_states:
						dfs(next_state.name, path + [symbol])

		dfs(state.name, [])
		return tails

	def k_tails(self, xs, k=2, verbose=0):
		automata = self.build_pta(xs)
		return self.merge_states(automata, k)





