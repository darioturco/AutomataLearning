from src.L import LStartLearner
from src.automatas.automata_learning import Learner as AutomataLearner
from experiments.performance_automatas import measure_automata_performance_in_functions
from tests.automatas.test_functions import problem1


if __name__ == "__main__":
	#measure_transducer_performance_in_datasets()


	#learner = TransducerLearner(16, alphabet_in_6, alphabet_out_6)
	#transducer = learner.learn_from_dataset(xs6, ys6, run_n=10, verbose=0)
	#print(f"    Square Error Sum: {transducer.error_square(xs6, ys6)}\n")


	#learner = AutomataLearner(2, ['0', '1'])
	#xs = ['011101', '01010', '111111', '11']
	#ys = ['010101', '01010', '010101', '01']
	#automata = learner.learn_from_dataset(xs, ys, run_n=256, verbose=0)
	#print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	#automata_s = automata.to_state_automata()
	#automata_s.print()

	##### Performance experiment #####
	res = measure_automata_performance_in_functions(pr=0.75, le=10, run_n=2500, train_step_n=3000, save=False)
	automatas, train_errors, test_errors, times = res
	automatas[0].show()

	##### L Start #####


	#learner = LStartLearner(problem1.alphabet)
	#automata = learner.learn(problem1.target_automata)
	#automata.show()

	# print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	# automata_s = automata.to_state_automata()
	# automata_s.print()






