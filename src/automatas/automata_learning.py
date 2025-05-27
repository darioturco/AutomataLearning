from src.automatas.algorithms.k_tails import KTail
from src.automatas.algorithms.rpni import RPNI
from src.automatas.algorithms.edsm import EDSM
from src.automatas.algorithms.derivative_learner import DerivativeLearner

class Learner:
	def __init__(self, alphabet, seed=42, verbose=0):
		self.alphabet = alphabet
		self.verbose = verbose
		self.seed = seed

	def derivative_passive_learn(self, xs, ys, max_states, concatenate=False, lazy_bias=1.0, train_step_n=1000, run_n=1000, learning_rate=0.25, batch_size=10, b1=0.5, b2=0.5):
		dl = DerivativeLearner(self.alphabet, max_states, lazy_bias=lazy_bias, train_step_n=train_step_n, run_n=run_n, learning_rate=learning_rate, batch_size=batch_size, b1=b1, b2=b2, seed=self.seed, verbose=self.verbose)
		return dl.learn_from_dataset(xs, ys, concatenate)

	def derivative_active_learn(self, target_automata, budget, t, p, max_states, concatenate=False, lazy_bias=1.0, train_step_n=1000, run_n=1000, learning_rate=0.25, batch_size=10, b1=0.5, b2=0.5):
		dl = DerivativeLearner(self.alphabet, max_states, lazy_bias=lazy_bias, train_step_n=train_step_n, run_n=run_n, learning_rate=learning_rate, batch_size=batch_size, b1=b1, b2=b2, seed=self.seed, verbose=self.verbose)
		return dl.learn(target_automata, budget, t, p, concatenate=concatenate)

	def learn_from_k_tail(self, xs, k, verbose=0):
		return KTail(self).learn(xs, k, verbose=verbose)

	def learn_from_rpni(self, xs, ys, verbose=0):
		return RPNI(self).learn_rpni(xs, ys, verbose=verbose)

	def learn_from_edsm(self, xs, ys, verbose=0):
		return EDSM(self).learn_edsm(xs, ys, verbose=verbose)
