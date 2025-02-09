from tests.tranducers.test_dataset import *
from tests.automatas.test_functions import *
from src.transducers.transducer_learning import Learner as TransducerLearner
from src.automatas.automata_learning import Learner as AutomataLearner
from src.utils import sample_dataset
import time
import numpy as np


def measure_automata_performance_in_functions(pr=0.75, le=10, run_n=1000, entropy_weight=0, lazy_bias=1.0, train_step_n=1000, learning_rate=0.1, b1=0.4, b2=0.4, problem_list=None, save=False):
    all_problems = [problem1, problem2, problem3, problem4, problem5, problem6, problem7, problem8, problem9, problem10, problem11, problem12, problem13, problem14, problem15, problem16]
    if problem_list is None:
        problems = all_problems
    else:
        problem_list = np.array(problem_list) - 1
        problems = list(np.array(all_problems)[problem_list])

    automatas, train_errors, test_errors, times = {}, {}, {}, {}
    for i, p in enumerate(problems):
        learner = AutomataLearner(p.max_states, p.alphabet, entropy_weight=entropy_weight, lazy_bias=lazy_bias, train_step_n=train_step_n, learning_rate=learning_rate, b1=b1, b2=b2)
        xs, ys = sample_dataset(p.f, p.alphabet, pr, le)
        test_xs, test_ys = sample_dataset(p.f, p.alphabet, pr, le)
        xs = xs + p.xs
        ys = ys + p.ys
        print(f"Problem{p.description}")
        print(f"    xs: {xs}")
        print(f"    ys: {ys}")


        start_time = time.time() # Time start
        automata = learner.learn_from_dataset(xs, ys, run_n)
        end_time = time.time() # Time end

        times[p.num] = end_time - start_time
        train_errors[p.num] =automata.error_square(xs, ys)
        test_errors[p.num] = automata.error_square(test_xs, test_ys)
        automatas[p.num] = automata

        print(f"    Time: {times[p.num]:.2f} seg")
        print(f"    Train Square Error Sum: {train_errors[p.num]}")
        print(f"    Test Square Error Sum: {test_errors[p.num]}\n")

    ### si save es true guardar los resultados en un vsc y los plots de los automatas

    return automatas, train_errors, test_errors, times

def measure_transducer_performance_in_datasets():
    xss = [xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10]
    yss = [ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10]
    alphabets_in = [alphabet_in_1, alphabet_in_2, alphabet_in_3, alphabet_in_4, alphabet_in_5, alphabet_in_6, alphabet_in_7, alphabet_in_8, alphabet_in_9, alphabet_in_10]
    alphabets_out = [alphabet_out_1, alphabet_out_2, alphabet_out_3, alphabet_out_4, alphabet_out_5, alphabet_out_6, alphabet_out_7, alphabet_out_8, alphabet_out_9, alphabet_out_10]

    transducers = []
    for i, (xs, ys, alphabet_in, alphabet_out) in enumerate(zip(xss, yss, alphabets_in, alphabets_out)):
        learner = TransducerLearner(10, alphabet_in, alphabet_out)
        print(f"Problem {i+1}:")
        print(f"    xs: {xs}")
        print(f"    ys: {ys}")
        transducer = learner.learn_from_dataset(xs, ys, alphabet_in, alphabet_out)
        print(f"    Square Error Sum: {transducer.error_square(xs, ys)}\n")
        transducers.append(transducer)

    return transducers



