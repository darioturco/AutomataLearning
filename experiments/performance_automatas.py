from tests.automatas.test_functions import *
from src.automatas.automata_learning import Learner as AutomataLearner
from src.automatas.automatas import StateAutomata
from src.utils import sample_dataset, save_pickle
from datetime import datetime
import time
import numpy as np
import random

# Default parameters
pr=0.75
le=10
run_n=900
concatenate=False
lazy_bias=1.0
train_step_n=1000
learning_rate=0.05
batch_size = 10
b1=0.4
b2=0.4
seed=42

np.random.seed(seed)

def filter_positives(xs, ys):
    return [x for x, y in zip(xs, ys) if len(y) > 0 and y[-1] == '1']

def get_problems(problem_list):
    all_problems = [problem1, problem2, problem3, problem4, problem5, problem6, problem7, problem8, problem9, problem10,
                    problem11, problem12, problem13, problem14, problem15, problem16]
    if type(problem_list) == dict:
        problems_n = problem_list["problems"]
        dataset_size = problem_list["dataset_size"]
        max_n_states = problem_list["max_n_states"]
        end_p = problem_list["end_p"]
        n_transitions = problem_list["n_transitions"]
        problems = []
        for i in range(problems_n):
            dfa = StateAutomata.generate_random_dfa(['a', 'b'], max_n_states, n_transitions, end_p)
            xs, _ = sample_dataset(dfa.accept, dfa.alphabet, pr, dataset_size)
            problems.append(Problem(dfa, xs, f"DFA {i}", i))

    elif len(problem_list) == 0:
        problems = all_problems
    else:
        problem_list = np.array(problem_list) - 1
        problems = list(np.array(all_problems)[problem_list])
    return problems

def get_algorithms():
    return {#'edsm': lambda alp, xs, ys, max_states, lr, batch_s: AutomataLearner(alp).learn_from_edsm(xs, ys),
            #'rpni': lambda alp, xs, ys, max_states, lr, batch_s: AutomataLearner(alp).learn_from_rpni(xs, ys),
            #'ktails': lambda alp, xs, ys, max_states, lr, batch_s: AutomataLearner(alp).learn_from_k_tail(filter_positives(xs, ys), k=None),
            'derivative': lambda alp, xs, ys, max_states, lr, batch_s: AutomataLearner(alp).derivative_passive_learn(xs, ys, max_states=max_states, concatenate=concatenate, run_n=run_n, lazy_bias=lazy_bias, train_step_n=train_step_n, learning_rate=lr, batch_size=batch_s, b1=b1, b2=b2)
            }

def uptime_hyperparameters(save=True):
    i = 0
    repetitions = 5
    f = get_algorithms()['derivative']
    for dataset_size in [10, 25, 50, 100]:
        for lr in [0.1, 0.05, 0.01, 0.005, 0.001]:
            for batch_s in [10, 20, 40, 60, 80, 100]:
                if batch_s > dataset_size:
                    continue

                times = {}
                train_errors = {}
                test_errors = {}
                automatas = {}
                problems = {}
                datasets = {}

                for r in range(repetitions):
                    max_n_states = 20 #random.randint(5, 50)
                    n_transitions = random.randint(2, max_n_states//2 + 1)
                    end_p = random.uniform(0.2, 0.5)

                    dfa = StateAutomata.generate_random_dfa(['a', 'b'], max_n_states, n_transitions, end_p)
                    xs, _ = sample_dataset(dfa.accept, dfa.alphabet, pr, dataset_size)
                    problem = Problem(dfa, xs, f"DFA {r}: lr = {lr} - batch_size = {batch_s} - dataset_size = {dataset_size}", i)
                    xs, ys = problem.xs, problem.ys
                    test_xs, test_ys = sample_dataset(problem.f, problem.alphabet, pr, dataset_size)

                    print(f"\nProblem {problem.description}")
                    print(f"    xs: {xs}")
                    print(f"    ys: {ys}")

                    start_time = time.time()  # Time start
                    try:
                        automata = f(problem.alphabet, xs, ys, problem.max_states, lr, batch_s)
                        end_time = time.time()  # Time end

                        problems[r] = problem
                        times[r] = end_time - start_time
                        train_errors[r] = automata.error_square(xs, ys)
                        test_errors[r] = automata.error_square(test_xs, test_ys)
                        automatas[r] = automata
                        datasets[r] = (xs, ys, test_xs, test_ys)

                        print(f"    Time: {times[r]:.2f} seg")
                        print(f"    Train Square Error Sum: {train_errors[r]}")
                        print(f"    Test Square Error Sum: {test_errors[r]}")

                    except Exception as e:
                        print(f"Error in: {problem.description} as {str(e)}")

                    i += 1

                if save:
                    name_file = f"{dataset_size}-{lr}-{batch_s}_"
                    res = {'name': name_file,
                        'automatas': automatas,
                        'train_errors': train_errors,
                        'test_errors': test_errors,
                        'times': times,
                        'problems': problems,
                        'datasets': datasets,
                        'hyperparameters': {"dataset_size": dataset_size, "learning_rate": lr, "batch_size": batch_s}}

                    date = datetime.now().strftime("%Y-%m-%d %H-%M")
                    folder = "hyperparameters"

                    # Save the hyperparameters used in this run and results
                    save_pickle(res, f'./experiments/{folder}/{name_file}{date}.pkl')
                    save_hyperparameters(f'./experiments/{folder}/hyper_{name_file}{date}.txt', {"learning_rate": lr, "batch_size": batch_s})

                    print(f"Saved {name_file}{date}.pkl")





def run_for_multiple_dfa_types(save=False):
    for dataset_size in [10, 50, 100, 200]:
    #for dataset_size in [50, 100, 200]:
    #for dataset_size in [200]:
        for max_n_states in [5, 10, 20, 30, 40, 50, 60, 70]:
            pl = {"dataset_size": dataset_size,
                  "max_n_states": max_n_states,
                  "n_transitions": 5,
                  "problems": 9, "end_p": 0.2}

            name_file = f"{dataset_size}_{max_n_states}"
            print(f"Running algorithms with dataset_size: {dataset_size} - max_n_states {max_n_states}")
            compare_passive_algorithms(pl, save=save, name_file=name_file)

""" Compare diferents algorithms solving the same problems. The problems to solve are in problem_list, if problem_list is None solve all the problems 
    and if problem_list is an integer number solve amount of random problems expressed by the number """
def compare_passive_algorithms(problem_list=None, save=False, name_file=""):
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
        print(f"\nProblem {p.description}")
        print(f"    xs: {xs}")
        print(f"    ys: {ys}")
        datasets[p.num] = (xs, ys)

        for a, f in algorithms.items():
            print(f"\n   Algorithm: {a}")
            start_time = time.time() # Time start
            automata = f(p.alphabet, xs, ys, p.max_states, learning_rate, batch_size)
            end_time = time.time() # Time end

            times[a][p.num] = end_time - start_time
            train_errors[a][p.num] = automata.error_square(xs, ys)
            test_errors[a][p.num] = automata.error_square(test_xs, test_ys)
            automatas[a][p.num] = automata

            print(f"    Time: {times[a][p.num]:.2f} seg")
            print(f"    Train Square Error Sum: {train_errors[a][p.num]}")
            print(f"    Test Square Error Sum: {test_errors[a][p.num]}")

    res = {'name': name_file,
           'problem_list': problem_list,
           'automatas': automatas,
           'train_errors': train_errors,
           'test_errors': test_errors,
           'times': times,
           'problems': problems,
           'datasets': datasets}

    if save:
        date = datetime.now().strftime("%Y-%m-%d %H-%M")
        if name_file != "":
            name_file += "_"
        folder = "pickle_obj"
        #folder = "tranditional"

        # Save the hyperparameters used in this run and results
        save_pickle(res, f'./experiments/{folder}/{name_file}{date}.pkl')
        save_hyperparameters(f'./experiments/{folder}/hyper_{name_file}{date}.txt', problem_list)

        print(f"Saved {name_file}{date}.pkl")

    return res


def save_hyperparameters(path, to_update):
    d = {"pr": pr,
         "le": le,
         "run_n": run_n,
         "concatenate": concatenate,
         "lazy_bias": lazy_bias,
         "train_step_n": train_step_n,
         "learning_rate": learning_rate,
         "batch_size": batch_size,
         "b1": b1,
         "b2": b2}

    if type(to_update) == dict:
        d.update(to_update)

    with open(path, 'w') as f:
        f.write(str(d))



