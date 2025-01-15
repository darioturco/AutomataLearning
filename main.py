from collections import defaultdict
#from src.L import *
from src.ALearning import Learner, TensorTransducer, FunctionTransducer
from tests.test_tensors import *
from tests.test_functions import *
from tests.test_dataset import *

if __name__ == "__main__":

        

        

	alphabet = ['0', '1']


	#target_fsm = TensorTransducer(T1, R1, s1, alphabet)
	target_fsm = FunctionTransducer(f1, alphabet)

	learner = Learner(5)
	#transducer = learner.learn(1000, verbose=1)

	
	transducer = learner.learn_from_dataset(xs1, ys1, alphabet, 1000, verbose=1)
	transducer.show(verbose=1)
