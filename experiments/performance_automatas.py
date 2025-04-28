from tests.automatas.test_functions import *
from src.automatas.automata_learning import Learner as AutomataLearner
from src.automatas.automatas import StateAutomata
from src.utils import sample_dataset, save_pickle
from datetime import datetime
import time
import numpy as np

import pickle

# Default parameters
pr=0.75
le=10
run_n=1000
concatenate=False
entropy_weight=0
lazy_bias=1.0
train_step_n=1000
learning_rate=0.1
b1=0.4
b2=0.4
examples = 32
max_n_states = 16
end_p = 0.2
nt = 3


def filter_positives(xs, ys):
    return [x for x, y in zip(xs, ys) if len(y) > 0 and y[-1] == '1']
def get_problems(problem_list):
    all_problems = [problem1, problem2, problem3, problem4, problem5, problem6, problem7, problem8, problem9, problem10,
                    problem11, problem12, problem13, problem14, problem15, problem16]
    if type(problem_list) == int:
        problems = []
        for i in range(problem_list):
            nfa = StateAutomata.generate_random_nfa(['a', 'b'], max_n_states, nt, end_p)
            xs, _ = sample_dataset(nfa.run_fsm, nfa.alphabet, pr, le)
            problems.append(Problem(nfa, xs, f"NFA {i}", i))

    elif len(problem_list) == 0:
        problems = all_problems
    else:
        problem_list = np.array(problem_list) - 1
        problems = list(np.array(all_problems)[problem_list])
    return problems

def get_algorithms():
    return {'rpni': lambda alp, xs, ys, max_states: AutomataLearner(alp).learn_from_rpni(xs, ys),
            'ktails': lambda alp, xs, ys, max_states: AutomataLearner(alp).learn_from_k_tail(filter_positives(xs, ys), k=None),
            'derivative': lambda alp, xs, ys, max_states: AutomataLearner(alp).derivative_passive_learn(xs, ys, max_states=max_states, concatenate=concatenate, run_n=run_n, entropy_weight=entropy_weight, lazy_bias=lazy_bias, train_step_n=train_step_n, learning_rate=learning_rate, b1=b1, b2=b2)}

""" Compare diferenst algoritms solving the same problems. The problems to solve are in problem_list, if problem_list is None solve all the problems 
    and if problem_list is an integer number solve amount of random problems expressed by the number """
def measure_automata_performance_in_functions(problem_list=None, save=False):
    problems = get_problems(problem_list)
    algorithms = get_algorithms()

    automatas = {n:{} for n in algorithms.keys()}
    train_errors = {n:{} for n in algorithms.keys()}
    test_errors = {n:{} for n in algorithms.keys()}
    times = {n:{} for n in algorithms.keys()}
    datasets = {}
    for i, p in enumerate(problems):
        xs, ys = sample_dataset(p.f, p.alphabet, pr, le)
        test_xs, test_ys = sample_dataset(p.f, p.alphabet, pr, le)
        xs = xs + p.xs
        ys = ys + p.ys
        print(f"\nProblem{p.description}")
        print(f"    xs: {xs}")
        print(f"    ys: {ys}")
        datasets[p.num] = (xs, ys)

        for a, f in algorithms.items():
            print(f"\n   Algorithm: {a}")
            start_time = time.time() # Time start
            automata = f(p.alphabet, xs, ys, p.max_states)
            end_time = time.time() # Time end

            times[a][p.num] = end_time - start_time
            train_errors[a][p.num] = automata.error_square(xs, ys)
            test_errors[a][p.num] = automata.error_square(test_xs, test_ys)
            automatas[a][p.num] = automata

            print(f"    Time: {times[a][p.num]:.2f} seg")
            print(f"    Train Square Error Sum: {train_errors[a][p.num]}")
            print(f"    Test Square Error Sum: {test_errors[a][p.num]}")

    res = {'automatas': automatas,
           'train_errors': train_errors,
           'test_errors': test_errors,
           'times': times,
           'problems': problems,
           'datasets': datasets}

    if save:
        date = datetime.now().strftime("%Y-%m-%d %H-%M")
        save_pickle(res, f'./experiments/pickle_obj/{date}.pkl')

        # Save the hyperparameters used in this run
        save_hyperparameters(f'./experiments/pickle_obj/hyper_{date}.txt')
        pass

    return res


def save_hyperparameters(path):
    d = {"pr": pr,
         "le": le,
         "run_n": run_n,
         "concatenate": concatenate,
         "entropy_weight": entropy_weight,
         "lazy_bias": lazy_bias,
         "train_step_n": train_step_n,
         "learning_rate": learning_rate,
         "b1": b1,
         "b2": b2}

    with open(path, 'w') as f:
        f.write(str(d))



