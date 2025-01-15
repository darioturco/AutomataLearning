from collections import defaultdict

class DFA:
    """Defines a deterministic finite automaton (for simulation and hypothesis)."""
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def accepts(self, string):
        """Simulates the DFA for a given string."""
        state = self.start_state
        for symbol in string:
            state = self.transitions.get((state, symbol))
            if state is None:
                return False
        return state in self.accept_states

class Teacher:
    """Defines the teacher/oracle with a target DFA."""
    def __init__(self, target_dfa):
        self.target_dfa = target_dfa

    def membership_query(self, string):
        """Checks if the string is in the target language."""
        return self.target_dfa.accepts(string)

    def equivalence_query(self, hypothesis_dfa):
        """Checks if the hypothesis DFA matches the target DFA."""
        # Compare the target DFA and hypothesis on all strings (up to a reasonable length).
        # This is simplified for educational purposes.
        max_length = 10
        for length in range(max_length + 1):
            for string in self.generate_strings(hypothesis_dfa.alphabet, length):
                if self.target_dfa.accepts(string) != hypothesis_dfa.accepts(string):
                    return False, string
        return True, None

    def generate_strings(self, alphabet, length):
        """Generates all strings over the given alphabet up to a certain length."""
        if length == 0:
            return [""]
        shorter_strings = self.generate_strings(alphabet, length - 1)
        return shorter_strings + [s + a for s in shorter_strings for a in alphabet]

class LStarLearner:
    """Implements the L* algorithm for DFA learning."""
    def __init__(self, alphabet, teacher):
        self.alphabet = alphabet
        self.teacher = teacher
        self.table = {"S": set(), "E": set(), "T": defaultdict(bool)}
        self.table["S"].add("")
        self.table["E"].add("")

    def update_table(self):
        """Populates the observation table using membership queries."""
        for s in self.table["S"]:
            for e in self.table["E"]:
                self.table["T"][(s, e)] = self.teacher.membership_query(s + e)

    def is_closed(self):
        """Checks if the table is closed."""
        for s in self.table["S"]:
            for a in self.alphabet:
                row = tuple(self.table["T"][(s + a, e)] for e in self.table["E"])
                if row not in [tuple(self.table["T"][(s_, e)] for e in self.table["E"]) for s_ in self.table["S"]]:
                    return False, s + a
        return True, None

    def is_consistent(self):
        """Checks if the table is consistent."""
        rows = {}
        for s in self.table["S"]:
            rows[s] = tuple(self.table["T"][(s, e)] for e in self.table["E"])
        for s1 in self.table["S"]:
            for s2 in self.table["S"]:
                if rows[s1] == rows[s2]:
                    for a in self.alphabet:
                        row1 = tuple(self.table["T"][(s1 + a, e)] for e in self.table["E"])
                        row2 = tuple(self.table["T"][(s2 + a, e)] for e in self.table["E"])
                        if row1 != row2:
                            return False, (s1, s2, a)
        return True, None

    def refine_table(self, counterexample):
        """Refines the observation table with a counterexample."""
        for i in range(len(counterexample) + 1):
            self.table["S"].add(counterexample[:i])
        self.update_table()

    def build_hypothesis(self):
        """Constructs a DFA from the observation table."""
        states = {tuple(self.table["T"][(s, e)] for e in self.table["E"]): i for i, s in enumerate(self.table["S"])}
        start_state = states[tuple(self.table["T"][("", e)] for e in self.table["E"])]
        accept_states = {states[tuple(self.table["T"][(s, e)] for e in self.table["E"])] for s in self.table["S"] if self.table["T"][(s, "")]}
        transitions = {}
        for s in self.table["S"]:
            for a in self.alphabet:
                row = tuple(self.table["T"][(s + a, e)] for e in self.table["E"])
                if row in states:
                    transitions[(states[tuple(self.table["T"][(s, e)] for e in self.table["E"])], a)] = states[row]
        return DFA(set(states.values()), self.alphabet, transitions, start_state, accept_states)

    def learn(self):
        """Main learning loop."""
        self.update_table()
        while True:
            # Ensure closure
            closed, witness = self.is_closed()
            if not closed:
                self.table["S"].add(witness)
                self.update_table()
                continue

            # Ensure consistency
            consistent, inconsistency = self.is_consistent()
            if not consistent:
                s1, s2, a = inconsistency
                self.table["E"].add(a)
                self.update_table()
                continue

            # Build hypothesis DFA
            hypothesis = self.build_hypothesis()
            equivalent, counterexample = self.teacher.equivalence_query(hypothesis)
            if equivalent:
                return hypothesis
            self.refine_table(counterexample)

