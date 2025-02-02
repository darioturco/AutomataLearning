from collections import defaultdict
#from src.L import *
from src.ALearning import Learner, TensorTransducer, FunctionTransducer
from tests.test_tensors import *
from tests.test_functions import *
from tests.test_dataset import *
from src.utils import generate_data_set, prepare_str, probabilistic_sample
from experiments.performance import measure_performance_in_datasets
import random




if __name__ == "__main__":
	measure_performance_in_datasets()