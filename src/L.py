from src.automatas.automatas import StateAutomata
from src.utils import lambda_char, probabilistic_sample

# Algorthm from "Learning regular sets from queries and counterexamples"
class LStartLearner:
    def __init__(self, alphabet, verbose=0):
        self.alphabet = alphabet
        self.verbose = verbose

        self.S = []
        self.E = []
        self.table = {}

    @staticmethod
    def contain_query(x, target_automata):
        return target_automata(x)[0]

    @staticmethod
    def equivalence_query(automata, target_automata, t=100, p=0.7):
        to_test = []
        for _ in range(t):
            test = probabilistic_sample(target_automata.alphabet, p=p)
            to_test.append(test)
            if target_automata.run_fsm(test)[0] != automata.run_fsm(test):
                return False, test

        return True, None

    def get_string_from_row(self, row):
        for s, v in self.table.items():
            if tuple(v) == row:
                if s == lambda_char:
                    return ""
                else:
                    return s

        return ""


    def build_automata(self):
        print("Building Atomata: ")
        print(self.table)

        states_t = list({tuple(v) for _, v in self.table.items()})
        states_dict = {st: i for i, st in enumerate(states_t)}
        states = [states_dict[st] for st in states_t]

        edges =  {}        # {s1: [(i, s_i), ...], s2:...}
        for st, i in states_dict.items():
            edges[i] = []
            for a in self.alphabet:
                new_row = self.get_string_from_row(st) + a
                if new_row in self.table:
                    state = states_dict[self.table[new_row]]
                    edges[i].append((a, state))
        initial_state = states_dict[tuple(self.table[lambda_char])]
        accepting_states = [states_dict[s] for s in states_t if s[0] == True]

        return StateAutomata(states, edges, accepting_states, initial_state, self.alphabet)

    def is_different_for_all_rows(self, row):
        for s in self.S:
            if tuple(row) == tuple(self.table[s]):
                return False
        return True

    def is_consistent(self):
        pass


    def is_closed(self):
        for s1 in self.S:
            if self.is_different_for_all_rows(s1):
                return False

        return True


    @staticmethod
    def concat(s1, s2, s3=""):
        if s1 == lambda_char:
            s1 = ''
        if s2 == lambda_char:
            s2 = ''
        if s3 == lambda_char:
            s3 = ''
        return s1+s2+s3

    def learn(self, target_automata, t=100, p=0.7):
        assert target_automata.alphabet == self.alphabet, "Error, alphabet should have the same length."

        self.E = {lambda_char}  # columns
        self.S = {lambda_char}.union(set(self.alphabet))  # rows
        self.table = {a: [self.contain_query(lambda_char, target_automata)] for a in self.alphabet + [lambda_char]}

        automata = self.build_automata()
        eq, example = self.equivalence_query(automata, target_automata, t, p)
        example = None

        while not eq:
            if example is not None:
                for i in range(len(example)):
                    self.S.add(example[:i+1])
                    self.table[example[:i+1]] = [self.contain_query(self.concat(example[:i+1], p), target_automata) for p in self.E]

            while not self.is_consistent() or self.is_closed():
                if not self.is_consistent():
                    for s1 in self.S:
                        for s2 in self.S:
                            for a in self.alphabet:
                                for e in self.E:
                                    if tuple(self.table[s1]) == tuple(self.table[s2]) and self.table[self.concat(s1,a,e)] != self.table[self.concat(s2,a,e)]:
                                        self.E.add(self.concat(s1,a))
                                        for r in self.S:
                                            self.table[r] = [self.contain_query(self.concat(r, p), target_automata) for p in self.E]

                if not self.is_closed():
                    for s1 in self.S:
                        for a in self.alphabet:
                            new_s = self.concat(s1, a)
                            if self.is_different_for_all_rows(new_s):
                                self.S.add(new_s)
                                self.table[new_s] = [self.contain_query(self.concat(new_s, p), target_automata) for p in self.E]

            automata = self.build_automata()
            eq, example = self.equivalence_query(automata, target_automata, t, p)

        return automata






















