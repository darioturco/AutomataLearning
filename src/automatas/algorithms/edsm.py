from src.automatas.automatas import StateAutomata
from aalpy.learning_algs import run_EDSM

class EDSM:
	def __init__(self, learner):
		self.learner = learner
		self.alphabet = self.learner.alphabet

	def learn_edsm(self, xs, ys, input_completeness='sink_state', verbose=0):
		data = set()
		set_alphabet = set(self.alphabet)
		for x, y in zip(xs, ys):
			assert set(x).issubset(set_alphabet)
			if len(x) == len(y):
				for i in range(len(x)):
					data.add((tuple(x[:i+1]), y[i] == '1'))
			else:
				data.add((tuple(x), y == '1'))

		dfa = run_EDSM(list(data), automaton_type='dfa', input_completeness=input_completeness, print_info=verbose)
		return StateAutomata.dfa_to_automata_state(dfa, self.alphabet)




