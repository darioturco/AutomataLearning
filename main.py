from collections import defaultdict
#from src.L import *
from src.ALearning import Learner, TensorTransducer, FunctionTransducer
from tests.test_tensors import *
from tests.test_functions import *
from tests.test_dataset import *
from src.utils import generate_data_set

if __name__ == "__main__":

	#target_fsm = TensorTransducer(T1, R1, s1, alphabet=['0', '1'])
	target_fsm = FunctionTransducer(f1, alphabet=['0', '1'])

	learner = Learner()
	#transducer = learner.learn(1000, verbose=1)

	#transducer = learner.learn_from_dataset(xs1, ys1, ['0', '1'], 1000, verbose=1)
	#transducer = learner.learn_from_dataset(xs2, ys2, ['0', '1', '2', '3', '4', '5'], 1000, verbose=1)
	#transducer = learner.learn_from_dataset(xs3, ys3, ['0', '1', '2', '3', '4', '5'], 1000, verbose=1)
	#transducer = learner.learn_from_dataset(xs4, ys4, ['a', 'b'], 1000, verbose=0)

	alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
	#xs, ys = generate_data_set(f1, alphabet=['0', '1'], records=16, max_length=10)
	xs, ys = generate_data_set(f2, alphabet=alphabet, records=16, min_length=2,  max_length=32)
	print(xs, ys)
	transducer = learner.learn_from_dataset(xs, ys, alphabet, 1000, state_max=12, verbose=0)

	print(f"Error: {transducer.error_square(xs, ys)}")

	#state_transducer = transducer.to_state_transducer()
	#state_transducer.print()
	#state_transducer.show(verbose=0)
	transducer.show(verbose=0)
