from src.L import LStartLearner
from src.transducers.transducer_learning import Learner as TransducerLearner
from src.automatas.automata_learning import Learner as AutomataLearner
from experiments.performance_automatas import measure_automata_performance_in_functions
from tests.automatas.test_functions import problem1


if __name__ == "__main__":
	#measure_transducer_performance_in_datasets()


	#learner = TransducerLearner(16, alphabet_in_6, alphabet_out_6, run_n=10)
	#transducer = learner.learn_from_dataset(xs6, ys6, verbose=0)
	#print(f"    Square Error Sum: {transducer.error_square(xs6, ys6)}\n")



	#learner = AutomataLearner(['0', '1'], verbose=0)
	#xs = ['011101', '01010', '111111', '11', '1010000100']
	#ys = ['010101', '01010', '010101', '01', '0101010101']
	#automata = learner.derivative_passive_learn(xs, ys, max_states=4, concatenate=False, run_n=256)
	#print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	#automata_s = automata.to_state_automata()
	#automata_s.print()
	#automata_s.show()

	#res = measure_automata_performance_in_functions(pr=0.75, le=10, run_n=500, train_step_n=1000, save=False)
	#automatas, train_errors, test_errors, times = res
	#automatas[1].show()



	##### K Tails #####

	learner = AutomataLearner(['a', 'b'], False)
	#xs = ['abaa', 'a', 'aaa', 'babaa', 'bbbabbb', 'babbabbba', 'aaaaa', 'bbbabaa']
	xs = ['abcf', 'abcbcf']
	#xs = ['bcbca', 'aabca', 'aabcbca', 'aaa']
	#xs = ['aba', 'aa', 'ba']


	automata = learner.learn_from_k_tail(xs, k=None, verbose=0)
	automata.show(verbose=1)








