#from src.L import *
from src.automatas.automata_learning import Learner as AutomataLearner
from experiments.performance_automatas import measure_automata_performance_in_functions


if __name__ == "__main__":
	#measure_transducer_performance_in_datasets()


	#learner = TransducerLearner(16, alphabet_in_6, alphabet_out_6)
	#transducer = learner.learn_from_dataset(xs6, ys6, run_n=10, verbose=0)
	#print(f"    Square Error Sum: {transducer.error_square(xs6, ys6)}\n")


	learner = AutomataLearner(6, ['0', '1'])
	xs = ['011101', '01010', '111111', '11']
	ys = ['010101', '01010', '010101', '01']
	automata = learner.learn_from_dataset(xs, ys, run_n=256, verbose=0)
	print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	automata_s = automata.to_state_automata()
	automata_s.print()
	automata_s.show()


	#res = measure_automata_performance_in_functions(pr=0.75, le=10, run_n=500, train_step_n=1000, save=False)
	#automatas, train_errors, test_errors, times = res

	#automatas[1].show()




