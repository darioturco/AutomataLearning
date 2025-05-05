from experiments.performance_automatas import run_for_multiple_dfa_types
from src.automatas.automata_learning import Learner as AutomataLearner
from src.transducers.transducer_learning import Learner as TransducerLearner

if __name__ == "__main__":
	learner = TransducerLearner(['a', 'b'], ['0', '1'], verbose=0)
	xs = ['abbbab', 'ababa', 'aaaaaa', 'aa', 'ababbbbabb']
	ys = ['010101', '01010', '010101', '01', '0101010101']
	transducer = learner.derivative_passive_learn(xs, ys, max_states=4, concatenate=False, run_n=100)
	print(f"    Square Error Sum: {transducer.error_square(xs, ys)}\n")
	transducer_s = transducer.to_state_transducer()
	transducer_s.print()
	transducer_s.show()




