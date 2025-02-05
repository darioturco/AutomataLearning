#from src.L import *
from src.automatas.automata_learning import Learner as AutomataLearner


def f(xs):
	res = []
	last = '0'
	for x in xs:
		if x == '1':
			last = '0' if last == '1' else '1'
		res.append(last)
	return res


if __name__ == "__main__":
	#measure_performance_in_datasets()


	#learner = TransducerLearner(16, alphabet_in_6, alphabet_out_6)
	#transducer = learner.learn_from_dataset(xs6, ys6, run_n=10, verbose=0)
	#print(f"    Square Error Sum: {transducer.error_square(xs6, ys6)}\n")


	learner = AutomataLearner(2, ['0', '1'])
	xs = ['011101', '01010', '111111', '11']
	ys = ['010101', '01010', '010101', '01']
	automata = learner.learn_from_dataset(xs, ys, run_n=256, verbose=0)
	print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	automata.print()