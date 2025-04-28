from src.L import LStartLearner
from src.transducers.transducer_learning import Learner as TransducerLearner
from src.automatas.automata_learning import Learner as AutomataLearner
from experiments.performance_automatas import measure_automata_performance_in_functions
from tests.automatas.test_functions import problem1


if __name__ == "__main__":
	learner = AutomataLearner(['0', '1'], verbose=0)
	xs = ['011101', '01010', '111111', '11', '1010000100']
	ys = ['010101', '01010', '010101', '01', '0101010101']
	automata = learner.derivative_passive_learn(xs, ys, max_states=4, concatenate=False, run_n=256)
	print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	automata_s = automata.to_state_automata()
	automata_s.print()
	automata_s.show()

	#res = measure_automata_performance_in_functions(problem_list=None, save=True)



	##### Compare all problems #####
	#measure_automata_performance_in_functions(problem_list=4, save=True)










