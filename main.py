from collections import defaultdict
#from src.L import *
from src.ALearning import Learner, TensorTransducer, FunctionTransducer
from tests.test_tensors import *
from tests.test_functions import *
from tests.test_dataset import *
from src.utils import generate_data_set, prepare_str


def error_square(xs, ys0, transducer):

	x = transducer.separate_char.join(xs) + transducer.separate_char
	y0 = transducer.separate_char.join(ys0) + transducer.separate_char
	x = prepare_str(x, transducer.alphabet_in_ext)
	y0 = prepare_str(y0, transducer.alphabet_out_ext)
	#error = 0.0
	#for x, y0 in zip(xs, ys0):
	fsm = transducer.fsm
	y, s = TensorTransducer.run_fsm_with_values(x, fsm.R, fsm.T, fsm.s0)
	return jnp.square(y - y0).sum()





if __name__ == "__main__":
	learner = Learner()
	
	problem = problem2
	xs, ys = generate_data_set(problem.f, alphabet_in=problem.alphabet_in, records=8, min_length=2,  max_length=10, verbose=1)
	transducer = learner.learn_from_dataset(xs, ys, problem.alphabet_in, problem.alphabet_out, run_n=16, state_max=8, verbose=0)
	

	#print(f"Error: {transducer.error_square(xs, ys)}")
	print(f"Error: {error_square(xs, ys, transducer)}")
	print("---------------------")
	#transducer.print()


	#transducer.show(verbose=0)

