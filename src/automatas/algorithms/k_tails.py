from src.automatas.automatas import StateAutomata

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
	def learn_rpni(self, positives, negatives, verbose=0):
		if len(positives) == 0:
			return StateAutomata.all_negative_automata(self.alphabet)

		if len(negatives) == 0:
			return StateAutomata.all_positive_automata(self.alphabet)

		pta = self.create_prefix_tree_acceptor(positives, negatives)

		### Completar RPNI y pasar a otro archivo separado

		red_states = {pta.initial_state}
		blue_states = {new_state for (a, new_state) in pta.edges[pta.initial_state]}

		while len(blue_states) > 0:
			mergible_states = not_eq_states = [], []
			for blue_state in blue_states:
				merged = False
				for red_state in red_states:
					if self.are_mergable(blue_state, red_state, pta):
						mergible_states.append((blue_state, red_state))
						merged = True

				if not merged:
					not_eq_states.append(blue_state)

			# Promote
			red_states.add(not_eq_states[-1])

			blue_states = {pta.get_edge(red_state, a) for red_state in red_states for a in self.alphabet} - red_states - {None}







		"""

		while len(blue_states) > 0:
			blue_state = blue_states.pop()
			merged = False

				# Try merging blue_state with each red_state
			for red_state in sorted(red_states):
				# Check if merging is consistent
				consistent = True

				# Check if merging causes any conflicts in acceptance
				if (blue_state in pta.accepting_states and red_state not in pta.accepting_states) or \
						(blue_state not in pta.accepting_states and red_state in pta.accepting_states):
					consistent = False

				if consistent:
					# Perform merge: replace all occurrences of blue_state with red_state
					for state in pta.states:
						pta.edges[state] = [(c, red_state) if next_state == blue_state else (c, next_state) for c, next_state in pta.edges[state]]


					# Update states and transitions
					if blue_state in pta.states:
						pta.states.remove(blue_state)
					if blue_state in pta.accepting_states:
						pta.accepting_states.remove(blue_state)

					merged = True
					break

			if not merged:
				red_states.add(blue_state)
				# Update Blue: Add children of the newly added Red state
				for symbol in pta.edges[blue_state]:
					child = pta.get_edge(blue_state, symbol)
					if child not in red_states:
						blue_states.add(child)


		"""


		return pta

	def are_mergable(self, q1, q2, automaton):
		pass


