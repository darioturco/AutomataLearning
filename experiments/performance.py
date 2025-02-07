
from tests.tranducers.test_dataset import *
from tests.automatas.test_functions import *
from src.transducers.transducer_learning import Learner as TranducerLearner
from src.automatas.automata_learning import Learner as AutomataLearner
from src.utils import sample_dataset


def measure_automata_performance_in_functions():
    problems = [problem1, problem2, problem3, problem4, problem5, problem6, problem7, problem8, problem9, problem10]

    automatas = []
    for i, p in enumerate(problems):
        learner = AutomataLearner(p.max_states, p.alphabet)
        xs, ys = sample_dataset(p.f, p.alphabet, 0.4, 10)
        xs = xs + p.xs
        ys = ys + p.ys
        print(f"Problem {i+1}:")
        print(f"    xs: {xs}")
        print(f"    ys: {ys}")
        automata = learner.learn_from_dataset(xs, ys)
        print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
        automatas.append(automata)

    return automatas

def measure_transducer_performance_in_datasets():
    xss = [xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10]
    yss = [ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10]
    alphabets_in = [alphabet_in_1, alphabet_in_2, alphabet_in_3, alphabet_in_4, alphabet_in_5, alphabet_in_6, alphabet_in_7, alphabet_in_8, alphabet_in_9, alphabet_in_10]
    alphabets_out = [alphabet_out_1, alphabet_out_2, alphabet_out_3, alphabet_out_4, alphabet_out_5, alphabet_out_6, alphabet_out_7, alphabet_out_8, alphabet_out_9, alphabet_out_10]

    transducers = []
    for i, (xs, ys, alphabet_in, alphabet_out) in enumerate(zip(xss, yss, alphabets_in, alphabets_out)):
        learner = TranducerLearner(10, alphabet_in, alphabet_out)
        print(f"Problem {i+1}:")
        print(f"    xs: {xs}")
        print(f"    ys: {ys}")
        transducer = learner.learn_from_dataset(xs, ys, alphabet_in, alphabet_out)
        print(f"    Square Error Sum: {transducer.error_square(xs, ys)}\n")
        transducers.append(transducer)

    return transducers



