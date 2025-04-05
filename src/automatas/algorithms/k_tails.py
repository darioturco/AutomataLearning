from collections import defaultdict
from src.automatas.automatas import StateAutomata


class KTail:
	def __init__(self, learner):
		self.learner = learner

	def k_tail_new(self, pta, xs):
		#max_k = max([len(x) for x in xs])
		n = len(pta.states)
		m = [[0 for _ in range(n)] for _ in range(n)]
		for i in range(n):
			for j in range(n):
				if pta.is_accepting_state(i) and pta.is_accepting_state(j):
					m[i][j] = 1


	def learn(self, xs, k=1, verbose=0):
		# Step 1: Create prefix tree acceptor (PTA)
		pta = self.create_prefix_tree_acceptor(xs)

		#self.k_tail_new(pta, xs)

		# Step 2: Initialize state equivalence classes based on k-tails
		state_classes = {}
		for state in pta.states:
			k_tails = self.compute_k_tails(pta, state, k)
			state_classes[state] = frozenset(k_tails)

		# Step 3: Merge states with equivalent k-tails
		merged = True
		while merged:
			merged = False

			# Create mapping from k-tails to states
			tails_to_states = defaultdict(list)
			for state, tails in state_classes.items():
				tails_to_states[tails].append(state)

			print(tails_to_states)

			# Try to merge states with same k-tails
			for tails, states in tails_to_states.items():
				if len(states) > 1:
					# Merge all states in this group into the first one
					representative = states[0]
					for state in states[1:]:
						if self.merge_states(pta, representative, state):
							merged = True
							# Update state_classes for the merged state
							for s in states:
								if s in state_classes:
									del state_classes[s]
							state_classes[representative] = tails
							break
					if merged:
						break

		return pta

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


	def compute_k_tails(self, pta, state, k):
		tails = set()

		def dfs(s, path):
			if len(path) == k:
				tails.add(tuple(path))
				return
			if pta.is_accepting_state(s) and len(path) > 0:
				tails.add(tuple(path))
				return

			for symbol, target in pta.edges[s]:
				dfs(target, path + [symbol])

		dfs(state, [])
		return tails


	def create_prefix_tree_acceptor(self, xs):
		automaton = StateAutomata([0], {0: []}, [], 0, self.learner.alphabet)
		state_counter = 1
		final_states = []

		for seq in xs:
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

			final_states.append(current_state)

		automaton.set_accepting_states(final_states)
		return automaton