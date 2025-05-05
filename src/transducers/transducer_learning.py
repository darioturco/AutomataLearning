from src.transducers.algorithms.derivative_learner import DerivativeLearner

class Learner:
	def __init__(self, alphabet_in, alphabet_out, verbose=0):
		self.alphabet_in = alphabet_in
		self.alphabet_out = alphabet_out
		self.verbose = verbose

	def derivative_passive_learn(self, xs, ys, max_states, concatenate=False, entropy_weight = 0, lazy_bias = 1.0, train_step_n = 1000, run_n = 1000, learning_rate = 0.25, b1 = 0.5, b2 = 0.5):
		dl = DerivativeLearner(self.alphabet_in, self.alphabet_out, max_states, entropy_weight=entropy_weight, lazy_bias=lazy_bias, train_step_n=train_step_n, run_n=run_n, learning_rate=learning_rate, b1=b1, b2=b2, verbose=self.verbose)
		return dl.learn_from_dataset(xs, ys, concatenate)

	def derivative_active_learn(self, target_automata, budget, t, p, max_states, concatenate=False, entropy_weight = 0, lazy_bias = 1.0, train_step_n = 1000, run_n = 1000, learning_rate = 0.25, b1 = 0.5, b2 = 0.5):
		dl = DerivativeLearner(self.alphabet_in, self.alphabet_out, max_states, entropy_weight=entropy_weight, lazy_bias=lazy_bias, train_step_n=train_step_n, run_n=run_n, learning_rate=learning_rate, b1=b1, b2=b2, verbose=self.verbose)
		return dl.learn(target_automata, budget, t, p, concatenate=concatenate)


