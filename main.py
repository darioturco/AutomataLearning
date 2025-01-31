from collections import defaultdict
#from src.L import *
from src.ALearning import Learner, TensorTransducer, FunctionTransducer
from tests.test_tensors import *
from tests.test_functions import *
from tests.test_dataset import *
from src.utils import generate_data_set, prepare_str, probabilistic_sample
import random




if __name__ == "__main__":
	learner = Learner()

	pre_save_problem = False
	if pre_save_problem:
		problem = problem2
		xs, ys = generate_data_set(problem.f, alphabet_in=problem.alphabet_in, records=8, min_length=2, max_length=10,
								   verbose=1)
	else:
		problem = Problem(None, ['I', 'V', 'X', 'L', 'C', 'D', 'M', '#'],
						  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#'])
		xs, ys = xs5, ys5
	transducer = learner.learn_from_dataset(xs, ys, problem.alphabet_in, problem.alphabet_out, run_n=500, state_max=16, verbose=0)

	print(f"Error sum: {transducer.error_square(xs, ys)}")
	# print(f"Error: {error_square(xs, ys, transducer)}")
	print(transducer.run_fsm("III"))
	print("---------------------")
# transducer.print()


# transducer.show(verbose=0)