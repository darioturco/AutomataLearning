from collections import namedtuple
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
import random
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


	### Fix entropy_weight (no todos los tipos de automatas pueden devolver una lista de estados)
	def error_square(self, xs, ys0, entropy_weight=0):
		error = 0.0
		for x, y0 in zip(xs, ys0):
			y, s = self.run_fsm([x])
			y0 = jnp.array([prepare_str(y0, ['0', '1', self.separate_char])])
			error += jnp.square(y-y0).sum()
			#error += 0.0 if s is None else entropy(s.mean(0)) * entropy_weight
		return error

	def __call__(self, x):
		raise NotImplementedError

	def run_fsm(self, inputs):
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
		# x.shape = [len(xs), len(alphabet)]
		# T.shape = [len(alphabet), len(states), len(states)]
		# A.shape = [len(alphabet), len(states)]
		def f(s, x):
			s1 = jnp.einsum('ix,is,xst->it', x, s, T)
			y  = jnp.einsum('is,sy->iy', s1, A)

			return s1, (y, s1)

		init_state = jnp.array([s0 for _ in range(inputs.shape[0])])
		_, (outputs, states) = jax.lax.scan(f, init_state, inputs.transpose((1, 0, 2)))
		return outputs.transpose(1, 0, 2), jnp.vstack([[init_state], states]).transpose((1,0,2))

	""" Returns the string corresponding to x """
	def __call__(self, x):
		y, _ = self.run_fsm([x])
		return decode_str(y, ['0', '1', self.separate_char])

	def run_fsm(self, inputs):
		inputs = jnp.array([prepare_str(x, self.alphabet_ext) for x in inputs])
		return TensorAutomata.run_fsm_with_values(inputs, self.fsm.A, self.fsm.T, self.fsm.s0)

	""" Return True if accept the string False if it doesn't """
	def accept(self, x):
		y = self(x)
		return y[-1] == '1'

	def __repr__(self):
		return f"""
T = {self.fsm.T.shape}
R = {self.fsm.A.shape}
Initial State = {self.fsm.s0.shape}
		"""

	def print(self):
		print(self.__repr__())

	def show(self, path=None, title="", node_size=500, verbose=0):
		state_automata = self.to_state_automata()
		return state_automata.show(path=path, title=title, node_size=node_size, verbose=verbose)

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
	def __init__(self, f, alphabet, max_states=8):
		self.f = f
		super().__init__(alphabet, max_states)	### Ver que hacer con ese max_states=8

	def __repr__(self):
		return self.__repr__()

	def __call__(self, x):
		return ["1" if self.accept(x[:i]) else '0' for i in range(1, len(x)+1)]

	def run_fsm(self, inputs):
		tensor_y = [self(i) for i in inputs]
		return jnp.array([prepare_str("".join(ty), ['0', '1', self.separate_char]) for ty in tensor_y]), None

	def accept(self, x):
		return self.f(x)
	
	def print(self):
		raise NotImplementedError
	
	def show(self, path=None, title="", node_size=500, verbose=0):
		raise NotImplementedError

	def to_state_automata(self):
		raise NotImplementedError

class StateAutomata(Automata):
	@staticmethod
	def all_negative_automata(alphabet):
		return StateAutomata([0], {0: [(a, 0) for a in alphabet]}, [], 0, alphabet)

	@staticmethod
	def all_positive_automata(alphabet):
		return StateAutomata([0], {0: [(a, 0) for a in alphabet]}, [0], 0, alphabet)

	@classmethod
	def dfa_to_automata_state(cls, dfa, alphabet):
		state_setup = dfa.to_state_setup()
		state_dict = {k: i for i, k in enumerate(state_setup.keys())}
		states = list(state_dict.values())
		edges_dict = {s: [] for s in states}
		for k, (_, d) in state_setup.items():
			for a, s2 in d.items():
				edges_dict[state_dict[k]].append((a, state_dict[s2]))

		accepting_states = [state_dict[k] for k, v in state_setup.items() if v[0]]
		return cls(states, edges_dict, accepting_states, state_dict[dfa.initial_state.state_id], alphabet)

	""" Genera un DFA con n estados con nt transiciones por cada estado, donde las transiciones
		   son aleatoreas y hay probabilidad end_p de que un estado sea final """
	@classmethod
	def generate_random_dfa(cls, alphabet, n, nt, end_p):
		states = [i for i in range(n)]

		t_per_state = min(nt, len(states), len(alphabet))
		edges_dict = {s1: [(a, s2) for a, s2 in zip(random.sample(alphabet, k=t_per_state), random.sample(states, k=t_per_state))] for s1 in states}
		accepting_states = [s for s in states if random.random() < end_p]
		if len(accepting_states) == 0:
			accepting_states.append(random.choice(states))
		return cls(states, edges_dict, accepting_states, 0, alphabet)

	def __init__(self, states, edges, accepting_states, initial_state, alphabet):
		super().__init__(alphabet, len(states)+1)
		self.states = list(set(states))
		self.edges = edges					# {s1: [(i, s_i), ...], s2:...}
		self.accepting_states = accepting_states
		self.initial_state = initial_state
		self.trap_state = -1
		self.actual_state = self.initial_state

	# Return the new state before consume 'input_' in some 'state'
	def next_state(self, state, input_):
		if state not in self.edges:
			return None
		if state == self.trap_state:
			return self.trap_state

		for (i, s) in self.edges[state]:
			if input_ == i:
				return s
			
		return None

	def __call__(self, x):
		return "".join(['1' if s in self.accepting_states else '0' for s in self.run_fsm([x])[1][0][1:]])

	def run_fsm(self, inputs):
		res = []
		states_res = []

		for x in inputs:
			states = [self.initial_state]
			acceptors = []
			for i in x:
				state = self.next_state(states[-1], i)  # [(i, s), ...]

				if state is None:
					state = self.trap_state

				states.append(state)
				acceptors.append('1' if state in self.accepting_states else '0')

			res.append(prepare_str("".join(acceptors), ['0', '1', self.separate_char]))
			states_res.append(states.copy())

		return jnp.array(res), states_res

	def accept(self, x):
		y = self(x)
		return y[-1] == '1'

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

	def __repr__(self):
		return f"""
States: {self.states}
Edges: {self.edges}
Accepting Sates: {self.accepting_states}
Initial State: {self.initial_state}
		"""

	def print(self):
		print(self)

	def show(self, view=True, name="Automata", verbose=0):
		if verbose:
			self.print()

		graph = graphviz.Digraph(name)
		for i, s in enumerate(self.states):
			shape = 'doublecircle' if s == self.initial_state else 'circle'
			color = 'green' if s in self.accepting_states else 'white'
			graph.node(str(s), shape=shape, color='black', fillcolor=color, style='filled')

		for s1, xs in self.edges.items():
			for c, s2 in xs:
				graph.edge(str(s1), str(s2), label=str(c))

		if view:
			graph.render(directory='graphs', view=True)
		return graph

	def to_state_automata(self):
		return self

	def add_state(self, state):
		self.states = list(set(self.states + [state]))
		self.edges[state] = []

	def add_transition(self, s1, c, s2):
		# assert s1 in self.states and s2 in self.states and c in self.alphabet ### Revisar

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

	def reset_states(self):
		states_map = {s:i for i, s in enumerate(self.states)}
		self.states = [s for s in range(len(self.states))]
		self.accepting_states = [states_map[a] for a in self.accepting_states]
		new_edges = {}
		for k, v in self.edges.items():
			new_edges[states_map[k]] = [(a, states_map[s]) for a, s in v]
		self.edges = new_edges
		self.initial_state = states_map[self.initial_state]

	def remove_state(self, state):
		self.states.remove(state)
		self.edges.pop(state)

		for k in self.edges.keys():
			self.edges[k] = [(c, node) for c, node in self.edges[k] if node != state]

		if state in self.accepting_states:
			self.accepting_states.remove(state)

	def step(self, x):
		new_state = self.next_state(self.actual_state, x)
		if new_state is None:
			new_state = self.trap_state
		self.actual_state = new_state
		return new_state

	def reset(self):
		self.actual_state = self.initial_state


