from tests.automatas.test_functions import *
from src.automatas.automata_learning import Learner as AutomataLearner
from src.utils import sample_dataset, get_separate_char
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



def measure_alphabet_impact(pr=0.75, le=10, run_n=1000, entropy_weight=0, lazy_bias=1.0, train_step_n=1000, learning_rate=0.1, b1=0.4, b2=0.4, max_alphabet_extend=30, save=False):
    problems = [problem1]

    automatas, train_errors, test_errors, times = {}, {}, {}, {}
    for i, p in enumerate(problems):
        alphabet = p.alphabet.copy()
        times[p.num], train_errors[p.num], test_errors[p.num], automatas[p.num] = [], [], [], []

        print(f"Problem{p.description}")
        for _ in range(max_alphabet_extend):
            alphabet += [get_separate_char(alphabet)]
            learner = AutomataLearner(p.max_states, alphabet, entropy_weight=entropy_weight, lazy_bias=lazy_bias,
                                      train_step_n=train_step_n, learning_rate=learning_rate, b1=b1, b2=b2)
            xs, ys = sample_dataset(p.f, p.alphabet, pr, le)
            test_xs, test_ys = sample_dataset(p.f, p.alphabet, pr, le)
            xs = xs + p.xs
            ys = ys + p.ys

            print(f"Alphabet: {alphabet}")
            print(f"    xs: {xs}")
            print(f"    ys: {ys}")

            start_time = time.time()  # Time start
            automata = learner.learn_from_dataset(xs, ys, run_n)
            end_time = time.time()  # Time end

            times[p.num].append(end_time - start_time)
            train_errors[p.num].append(automata.error_square(xs, ys))
            test_errors[p.num].append(automata.error_square(test_xs, test_ys))
            automatas[p.num].append(automata)

            print(f"    Time: {times[p.num][-1]:.2f} seg")
            print(f"    Train Square Error Sum: {train_errors[p.num][-1]}")
            print(f"    Test Square Error Sum: {test_errors[p.num][-1]}\n")

    ### si save es true guardar los resultados en un vsc y los plots de los automatas

    return automatas, train_errors, test_errors, times

def measure_max_states_impact(pr=0.75, le=10, run_n=1000, entropy_weight=0, lazy_bias=1.0, train_step_n=1000, learning_rate=0.1, b1=0.4, b2=0.4, max_states_extend=100, save=False):
    problems = [problem1]

    automatas, train_errors, test_errors, times = {}, {}, {}, {}
    for i, p in enumerate(problems):
        times[p.num], train_errors[p.num], test_errors[p.num], automatas[p.num] = [], [], [], []

        print(f"Problem{p.description}")
        for j in range(1, max_states_extend):
            learner = AutomataLearner(j, p.alphabet, entropy_weight=entropy_weight, lazy_bias=lazy_bias,
                                      train_step_n=train_step_n, learning_rate=learning_rate, b1=b1, b2=b2)
            xs, ys = sample_dataset(p.f, p.alphabet, pr, le)
            test_xs, test_ys = sample_dataset(p.f, p.alphabet, pr, le)
            xs = xs + p.xs
            ys = ys + p.ys

            print(f"Max States: {j}")
            print(f"    xs: {xs}")
            print(f"    ys: {ys}")

            start_time = time.time()  # Time start
            automata = learner.learn_from_dataset(xs, ys, run_n)
            end_time = time.time()  # Time end

            times[p.num].append(end_time - start_time)
            train_errors[p.num].append(automata.error_square(xs, ys))
            test_errors[p.num].append(automata.error_square(test_xs, test_ys))
            automatas[p.num].append(automata)

            print(f"    Time: {times[p.num][-1]:.2f} seg")
            print(f"    Train Square Error Sum: {train_errors[p.num][-1]}")
            print(f"    Test Square Error Sum: {test_errors[p.num][-1]}\n")

    ### si save es true guardar los resultados en un vsc y los plots de los automatas

    return automatas, train_errors, test_errors, times

