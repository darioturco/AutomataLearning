from src.automatas.automatas import StateAutomata
from aalpy.learning_algs import run_RPNI
class KTail:
	def __init__(self, learner):
		self.learner = learner
		self.alphabet = self.learner.alphabet

	def add_single_state(self, q, equivalence):
		for eq in equivalence:
			if q in eq:
				return

		equivalence.append({q})

	def equivalent_next(self, q1, q2, automata, equivalences):
		for a in self.alphabet:
			new_q1 = automata.get_edge(q1, a)
			new_q2 = automata.get_edge(q2, a)
			if not self.equivalent(new_q1, new_q2, equivalences):
				return False

		return True

	def equivalent(self, q1, q2, equivalences):
		if q1 is None and q2 is None:
			return True

		for eq in equivalences:
			if (q1 in eq) and (q2 in eq):
				return True

		return False


	def open_equivalences(self, equivalence, all_equivalences, automaton):
		new_classes = []
		seen = []
		for q1 in equivalence:
			added = False
			for eq in new_classes:
				if q1 not in eq and self.equivalent_next(q1, next(iter(eq)), automaton, all_equivalences):
					eq.add(q1)
					added = True

			if not added:
				new_classes.append({q1})

		return new_classes


	def merge_states(self, automaton, state1, state2):
		### Check if the new states are well merged (if the transition are deleted or overwhiten)
		# Check if merge would create nondeterminism
		for symbol, target in automaton.edges[state2]:
			s = automaton.get_edge(state1, symbol)
			if s is not None and s != target:
				return False

		# Redirect all transitions to state2 to point to state1 instead
		for state in automaton.states:
			for symbol, target in automaton.edges[state]:
				if target == state2:
					automaton.add_transition(state, symbol, state1)

		# Copy transitions from state2 to state1
		for symbol, target in automaton.edges[state2]:
			s = automaton.get_edge(state1, symbol)
			if s is None:
				automaton.add_transition(state1, symbol, target)

		if state2 in automaton.accepting_states:
			automaton.add_accepting_state(state1)

		automaton.remove_state(state2)


	def create_prefix_tree_acceptor(self, positives, negatives):
		automaton = StateAutomata([0], {0: []}, [], 0, self.learner.alphabet)
		state_counter = 1
		final_states = []

		for seq in positives + negatives:
			current_state = automaton.initial_state
			for c in seq:
				s = automaton.get_edge(current_state, c)
				if s is None:
					new_state = state_counter
					state_counter += 1
					automaton.add_state(new_state)
					automaton.add_transition(current_state, c, new_state)
					current_state = new_state
				else:
					current_state = s
			if seq in positives:
				final_states.append(current_state)

		automaton.set_accepting_states(final_states)
		return automaton


	def learn(self, xs, k=None, verbose=0):
		if len(xs) == 0:
			return StateAutomata.all_negative_automata(self.alphabet)

		# Step 1: Create prefix tree acceptor (PTA)
		pta = self.create_prefix_tree_acceptor(xs, [])

		if k is None:
			k = len(pta.states)

		# Step 2: Compute the equivalences classes
		equivalences = [set(pta.accepting_states.copy()), {s for s in pta.states if s not in pta.accepting_states}]
		for i in range(k-1):
			new_equivalences = []
			for eq in equivalences:
				new_equivalences += self.open_equivalences(eq, equivalences, pta)
			equivalences = new_equivalences

		# Step 3: Merge all the states with the same equivalence class
		for eq in equivalences:
			q1 = eq.pop()
			for q2 in eq:
				self.merge_states(pta, q1, q2)

		pta.reset_states()
		return pta

	######### -------------------- #########
	def learn_rpni(self, xs, ys, input_completeness='sink_state', verbose=0):
		data = []
		set_alphabet = set(self.alphabet)
		for x, y in zip(xs, ys):
			assert set(x).issubset(set_alphabet)
			data.append((tuple(x), y == '1'))

		dfa = run_RPNI(data, automaton_type='dfa', input_completeness=input_completeness, print_info=verbose)
		return StateAutomata.dfa_to_automata_state(dfa, self.alphabet)




