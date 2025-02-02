from collections import defaultdict
#from src.L import *
from src.automatas.automata_learning import Learner as AutomataLearner
from src.transducers.transducer_learning import Learner as TransducerLearner
from tests.test_tensors import *
from tests.test_functions import *
from tests.test_dataset import *
from src.utils import generate_data_set, prepare_str, probabilistic_sample
from experiments.performance import measure_performance_in_datasets
import random




if __name__ == "__main__":
	#measure_performance_in_datasets()

	learner = TransducerLearner()
	transducer = learner.learn_from_dataset(xs6, ys6, alphabet_in_6, alphabet_out_6, run_n=500, state_max=16, verbose=0)
	print(f"    Square Error Sum: {transducer.error_square(xs6, ys6)}\n")