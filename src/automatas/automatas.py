from collections import namedtuple
from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from src.utils import decode_fsm, entropy, prepare_str, get_separate_char, decode_str, cartesian_product

FSM = namedtuple('FSM', 'T A s0')
Params = namedtuple('Params', 'T A s0')
Stats = namedtuple('Stats', 'total error entropy states_used')
TrainState = namedtuple('TrainState', 'params opt_state')
TrainResult = namedtuple('TrainResult', 'params eval logs')

class Automata:
	def __init__(self, alphabet, max_state):
		self.alphabet = alphabet
		self.separate_char = get_separate_char(alphabet)
		self.alphabet_ext = alphabet + [self.separate_char]
		self.CHAR = len(alphabet) + 1
		self.STATE_MAX = max_state

	def error_square(self, xs, ys0, entropy_weight=0):
		error = 0.0
		for x, y0 in zip(xs, ys0):
			y, s = self(x)
			y0 = prepare_str(y0, ['0', '1', self.separate_char])
			error += jnp.square(y-y0).sum()
			error += 0.0 if s is None else entropy(s.mean(0)) * entropy_weight
		return error

	def __call__(self, inputs):
		raise NotImplementedError

	def to_state_automata(self):
		raise NotImplementedError

	# Transducer Operations
	def union(self, transducer):
		raise NotImplementedError

	def intersection(self, transducer):
		raise NotImplementedError

	def composition(self, transducer):
		raise NotImplementedError

	def concatenation(self, transducer):
		raise NotImplementedError
		

class TensorAutomata(Automata):
	def __init__(self, T, A, s0, alphabet, max_state=8):
		super().__init__(alphabet, max_state)
		self.fsm = Params(T, A, s0)

	@staticmethod
	def run_fsm_with_values(inputs, A, T, s0):
		def f(s, x):
			s1 = jnp.einsum('x,s,xst->t', x, s, T)
			y  = jnp.einsum('s,sy->y', s1, A)

			return s1, (y, s1)

		_, (outputs, states) = jax.lax.scan(f, s0, inputs)
		return outputs, jnp.vstack([s0, states])

	def __call__(self, inputs):
		inputs = prepare_str(inputs, self.alphabet_ext)
		return TensorAutomata.run_fsm_with_values(inputs, self.fsm.A, self.fsm.T, self.fsm.s0)

	def run_fsm(self, x):
		y, _ = self(x)
		return decode_str(y, ['0', '1', self.separate_char])

	### Cambiar
	def show_fsm_story(xx, yy, ss):
		G = Digraph(graph_attr={'rankdir':'LR'}, node_attr={'shape':'circle'})
		G.node(ss[0], penwidth='3px')
		edges = set(zip(xx, yy, ss[:-1], ss[1:]))

		for x, y, a, b in edges:
			G.edge(a, b, '%s/%s'%(x, y))
			if len(set(ss)) > 2:
				G.engine = 'circo'
		return G

	def print(self):
		print(f"T = {self.fsm.T.shape}")
		print(f"R = {self.fsm.A.shape}")
		print(f"Initial State = {self.fsm.s0.shape}")

	### Cambiar
	def get_edges_out(self, n):
		edges = []
		for i in range(self.fsm.R.shape[0]):
			for j in range(self.fsm.R.shape[2]):
				if self.fsm.R[i][n][j] > 0.1:
					edges.append((self.alphabet_in_ext[i], self.alphabet_out_ext[j]))
		return edges

	### Cambiar
	def iterate_state_io(self):
		for s1 in range(self.fsm.T.shape[1]):
			for s2 in range(self.fsm.T.shape[1]):
				for i in self.alphabet:
					yield (s1, s2, i)

	### Cambiar
	def show(self, title="", verbose=0):
		if verbose:
			self.print()

		G, _, _ = self.to_nx_digraph()

		initial_state = int(jnp.argmax(self.fsm.s0))
		pos = nx.circular_layout(G)
		color_map = ["green" if n == initial_state else "lightblue" for n in G.nodes]
		nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, arrowsize=16, font_size=8)

		# Draw edge labels
		edge_labels = nx.get_edge_attributes(G, 'label')
		nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

		# Show the plot
		plt.title(title)
		plt.show()

	### Cambiar
	def to_nx_digraph(self):
		edges = {}
		### Mejorar los for anidados con un zip
		for s1, s2, i in self.iterate_state_io():
			if self.fsm.T[self.alphabet.index(i)][s1][s2] > 0.01:
				if (s1, s2) in edges:
					edges[(s1, s2)].append(f"\n{i}")
				else:
					edges[(s1, s2)] = [f"{i}"]

		# Create a directed graph
		G = nx.DiGraph()

		initial_state = int(jnp.argmax(self.fsm.s0))
		for (s1, s2), edge in edges.items():
			G.add_node(s1)
			G.add_node(s2)
			G.add_edge(s1, s2, label="".join(edge))

		# Removes the not reachable nodes
		dfs = nx.dfs_preorder_nodes(G, initial_state)
		nodes_dfs = {n for n in dfs}
		nodes = [n for n in G.nodes]
		for n in nodes:
			if n not in nodes_dfs:
				G.remove_node(n)

		return G, G.nodes, edges

	### Cambiar
	def to_state_automata(self):
		G, _, edges = self.to_nx_digraph()
		states = {n:i for i, n in enumerate(G.nodes)}
		
		edges_dict = {}
		for (s1, s2), xs in edges.items():
			if s1 in states and s2 in states:
				if not states[s1] in edges_dict:
					edges_dict[states[s1]] = []

				for x in xs:
					# x = "{self.alphabet[i]}"
					i = x.replace("\n", "")
					edges_dict[states[s1]].append((i, states[s2]))

		initial_state = states[int(jnp.argmax(self.fsm.s0))]
		accepting_states = [s for s, a in enumerate(self.fsm.A) if int(a[1]) > 0.01]
		return StateAutomata(list(states.values()), edges_dict, accepting_states, initial_state, self.alphabet)




class FunctionAutomata(Automata):
	def __init__(self, f, alphabet):
		self.f = f
		super().__init__(alphabet, 8)	### Ver que hacer con ese max_states=8

	def __call__(self, inputs):
		return self.f(inputs), None

	def run_fsm(self, x):
		return self(x)
	
	def print(self):
		raise NotImplementedError
	
	def show(self, title="", verbose=0):
		raise NotImplementedError

	def to_state_automata(self):
		raise NotImplementedError

class StateAutomata(Automata):
	def __init__(self, states, edges, accepting_states, initial_state, alphabet):

		super().__init__(alphabet, len(states))
		self.states = states
		self.edges = edges					# {s1: [(i, s_i), ...], s2:...}
		self.accepting_states = accepting_states
		self.initial_state = initial_state

	# Return the new state before consume 'input_' in the state 'state'
	def get_edge(self, state, input_):
		for (i, s) in self.edges[state]:
			if input_ == i:
				return s
			
		return None, None


	def __call__(self, inputs):
		states = [self.initial_state]
		for i in inputs:
			state = self.get_edge(states[-1], i)	# [(i, s), ...]

			### Falta chequear if it None
			states.append(state)
			
		return states[-1] in self.accepting_states

	def run_fsm(self, x):
		return self(x)

	def print(self):
		print(f"States: {self.states}")
		print(f"Edges: {self.edges}")
		print(f"Accepting Sates: {self.accepting_states}")
		print(f"Initial State: {self.initial_state}")

	### Cambiar
	def show(self, title="", verbose=0):
		G = nx.DiGraph()

		edges_dict = {}
		for s1, edges in self.edges.items():
			for i, s2, o in edges:
				if (s1, s2) in edges_dict:
					
					edges_dict[(s1, s2)].append(f"\n{i}/{o}")
				else:
					edges_dict[(s1, s2)] = [f"{i}/{o}"]

		for (s1, s2), edge in edges_dict.items():
			G.add_node(s1)
			G.add_node(s2)
			G.add_edge(s1, s2, label="".join(edge))

		initial_state = self.initial_state
		pos = nx.circular_layout(G)
		color_map = ["green" if n == initial_state else "lightblue" for n in G.nodes]
		nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, arrowsize=16, font_size=8)

		# Draw edge labels
		edge_labels = nx.get_edge_attributes(G, 'label')
		nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

		# Show the plot
		plt.title(title)
		plt.show()

		return G, G.nodes, edges

	def to_state_automata(self):
		return self





