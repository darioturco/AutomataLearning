from collections import namedtuple
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from graphviz import Source

from src.utils import decode_fsm, entropy, prepare_str, get_separate_char, decode_str, cartesian_product

FSM = namedtuple('FSM', 'T A s0')
Params = namedtuple('Params', 'T A s0')
Stats = namedtuple('Stats', 'total error entropy states_used')
TrainState = namedtuple('TrainState', 'params opt_state')
TrainResult = namedtuple('TrainResult', 'params eval logs')

### Hay que revisar esto:
###  __call__ tiene que devolver (lista de outputs, lista de estados)
### run_fsm tiene que devolver un solo booleano
### despues hay que swapear estas dos funciones

class Automata:
	def __init__(self, alphabet, max_state):
		self.alphabet = alphabet
		self.separate_char = get_separate_char(alphabet)
		self.alphabet_ext = alphabet + [self.separate_char]
		self.char_n = len(alphabet) + 1
		self.max_state = max_state



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

	def run_fsm(self, x):
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

		return decode_str(y, ['0', '1', self.separate_char]) ### Tiene que devolver un booleano

	def print(self):
		print(f"T = {self.fsm.T.shape}")
		print(f"R = {self.fsm.A.shape}")
		print(f"Initial State = {self.fsm.s0.shape}")

	def show(self, path=None, title="", node_size=500, verbose=0):
		state_automata = self.to_state_automata()
		state_automata.show(path=None, title="", node_size=500, verbose=0)




	def to_nx_digraph(self):
		edges_dict = {}
		for s_ in range(self.max_state):
			for i, c in enumerate(self.alphabet):
				new_s = int(jnp.einsum('x,s,xst->t', jnp.eye(1, self.char_n, i)[0], jnp.eye(1, self.max_state, s_)[0], self.fsm.T).argmax())
				if (s_, new_s) in edges_dict:
					edges_dict[(s_, new_s)].append(f"\n{c}")
				else:
					edges_dict[(s_, new_s)] = [f"{c}"]

		# Create a directed graph
		G = nx.DiGraph()

		initial_state = int(jnp.argmax(self.fsm.s0))
		for (s1, s2), edge in edges_dict.items():
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

		return G, G.nodes, edges_dict

	def to_state_automata(self):
		G, _, edges = self.to_nx_digraph()
		states = {n:i for i, n in enumerate(G.nodes)}
		
		edges_dict = {}
		for (s1, s2), xs in edges.items():
			if s1 in states and s2 in states:
				if not states[s1] in edges_dict:
					edges_dict[states[s1]] = []

				for x in xs:
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
		return self(x)[0]
	
	def print(self):
		raise NotImplementedError
	
	def show(self, path=None, title="", node_size=500, verbose=0):
		raise NotImplementedError

	def to_state_automata(self):
		raise NotImplementedError

class StateAutomata(Automata):
	def __init__(self, states, edges, accepting_states, initial_state, alphabet):
		super().__init__(alphabet, len(states))
		self.states = list(set(states))
		self.edges = edges					# {s1: [(i, s_i), ...], s2:...}
		self.accepting_states = accepting_states
		self.initial_state = initial_state

	# Return the new state before consume 'input_' in the state 'state'
	def get_edge(self, state, input_):
		if state not in self.edges:
			return None
		for (i, s) in self.edges[state]:
			if input_ == i:
				return s
			
		return None


	def __call__(self, inputs):
		states = [self.initial_state]
		for i in inputs:
			state = self.get_edge(states[-1], i)	# [(i, s), ...]

			### Falta chequear if it None
			states.append(state)
			
		return states[-1] in self.accepting_states

	def run_fsm(self, x):
		return self(x)

	def to_nx_digraph(self):
		G = nx.DiGraph()

		edges_dict = {}
		for s1, edges in self.edges.items():
			for i, s2 in edges:
				if (s1, s2) in edges_dict:
					edges_dict[(s1, s2)].append(f"\n{i}")
				else:
					edges_dict[(s1, s2)] = [f"{i}"]

		for (s1, s2), edge in edges_dict.items():
			G.add_node(s1)
			G.add_node(s2)
			G.add_edge(s1, s2, label="".join(edge))

		return G, G.nodes, edges_dict

	def print(self):
		print(f"States: {self.states}")
		print(f"Edges: {self.edges}")
		print(f"Accepting Sates: {self.accepting_states}")
		print(f"Initial State: {self.initial_state}")

	def show(self, path=None, title="", node_size=500, verbose=0):
		if verbose:
			self.print()

		graph = graphviz.Digraph('Automata')

		for i, s in enumerate(self.states):
			shape = 'doublecircle' if s == self.initial_state else 'circle'
			color = 'green' if s in self.accepting_states else 'white'
			graph.node(str(s), shape=shape, color='black', fillcolor=color, style='filled')

		for s1, xs in self.edges.items():
			for c, s2 in xs:
				graph.edge(str(s1), str(s2), label=str(c))

		graph.render(directory='graphs', view=True)

	def to_state_automata(self):
		return self

	def add_state(self, state):
		self.states = list(set(self.states + [state]))
		self.edges[state] = []

	def add_transition(self, s1, c, s2):
		### Check if the states are in self.satte and if the character is valid by the alphabet
		if s1 not in self.edges:
			self.edges[s1] = []

		### Check that the edges is not present yet
		self.edges[s1].append((c, s2))

	def add_accepting_state(self, state):
		self.accepting_states = list(set(self.accepting_states + [state]))

	def set_accepting_states(self, accepting_states):
		self.accepting_states = accepting_states

	def is_accepting_state(self, state):
		return state in self.accepting_states


	def remove_state(self, state):
		self.states.remove(state)
		self.edges.pop(state)

		for k in self.edges.keys():
			self.edges[k] = [(c, node) for c, node in self.edges[k] if node != state]

		if state in self.accepting_states:
			self.accepting_states.remove(state)









