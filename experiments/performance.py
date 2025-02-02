
from tests.test_dataset import *
from src.ALearning import Learner, TensorTransducer, FunctionTransducer

def measure_performance_in_datasets():
    xss = [xs1, xs2, xs3, xs4, xs5, xs6, xs7]
    yss = [ys1, ys2, ys3, ys4, ys5, ys6, ys7]
    alphabets_in = [alphabet_in_1, alphabet_in_2, alphabet_in_3, alphabet_in_4, alphabet_in_5, alphabet_in_6, alphabet_in_7]
    alphabets_out = [alphabet_out_1, alphabet_out_2, alphabet_out_3, alphabet_out_4, alphabet_out_5, alphabet_out_6, alphabet_out_7]

    for i, (xs, ys, alphabet_in, alphabet_out) in enumerate(zip(xss, yss, alphabets_in, alphabets_out)):
        learner = Learner()
        transducer = learner.learn_from_dataset(xs, ys, alphabet_in, alphabet_out, run_n=500, state_max=16, verbose=0)
        print(f"Problem {i+1}:")
        print(f"    xs: {xs}")
        print(f"    ys: {ys}")
        print(f"    Square Error Sum: {transducer.error_square(xs, ys)}\n")


