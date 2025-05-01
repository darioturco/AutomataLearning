from experiments.performance_automatas import run_for_multiple_dfa_types
from src.automatas.automata_learning import Learner as AutomataLearner

if __name__ == "__main__":
	#learner = AutomataLearner(['a', 'b'], verbose=0)
	#xs = ['abbbab', 'ababa', 'aaaaaa', 'aa', 'ababbbbabb']
	#ys = ['010101', '01010', '010101', '01', '0101010101']
	#automata = learner.derivative_passive_learn(xs, ys, max_states=4, concatenate=False, run_n=100)
	#print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	#automata_s = automata.to_state_automata()
	#automata_s.print()
	#automata_s.show()

	#res = compare_passive_algorithms(problem_list={"dataset_size": 100,  "problems": 5, "end_p": 0.2,
	#											  "max_n_states": 64, "n_transitions": 2}, save=True)

	run_for_multiple_dfa_types(save=True)



	##### Compare all problems #####
	#measure_automata_performance_in_functions(problem_list=2, save=True)

	##### Testear PRNI #####
	# Segun el algoritmo deberia ajustarse perfectamente al conjunto de entrenamiento
	#dfa = AutomataLearner(['a', 'b']).learn_from_rpni(xs, ys)
	#print(f"    Square Error Sum: {dfa.error_square(xs, ys)}\n")
	#automata_s = dfa.to_state_automata()
	#automata_s.print()
	#automata_s.show()











