from collections import namedtuple
from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from src.utils import entropy, prepare_str, get_separate_char, decode_str, cartesian_product

FSM = namedtuple('FSM', 'T R s0')
Params = namedtuple('Params', 'T R s0')
Stats = namedtuple('Stats', 'total error entropy states_used')
TrainState = namedtuple('TrainState', 'params opt_state')
TrainResult = namedtuple('TrainResult', 'params eval logs')
class Transducer:
	def __init__(self, alphabet_in, alphabet_out, max_state):
		self.alphabet_in = alphabet_in
		self.alphabet_out = alphabet_out
		self.separate_char = get_separate_char(alphabet_in + alphabet_out)
		self.alphabet_in_ext = alphabet_in + [self.separate_char]
		self.alphabet_out_ext = alphabet_out + [self.separate_char]
		self.char_n_in = len(alphabet_in) + 1
		self.char_n_out = len(alphabet_out) + 1
		self.max_state = max_state
		
	def error_square(self, xs, ys0, entropy_weight=0):
		error = 0.0
		for x, y0 in zip(xs, ys0):
			y, s = self(x)
			y0 = prepare_str(y0, self.alphabet_out_ext)
			error += jnp.square(y-y0).sum()
			error += 0.0 if s is None else entropy(s.mean(0)) * entropy_weight
		return error

	def cancat_error_square(self, xs, ys0):
		x = self.separate_char.join(xs) + self.separate_char
		y0 = self.separate_char.join(ys0) + self.separate_char
		x = prepare_str(x, self.alphabet_in_ext)
		y0 = prepare_str(y0, self.alphabet_out_ext)
		fsm = self.fsm
		y, s = TensorTransducer.run_fsm_with_values(x, fsm.R, fsm.T, fsm.s0)
		return jnp.square(y - y0).sum()

	def __call__(self, inputs):
		raise NotImplementedError

	def to_state_transducer(self):
		raise NotImplementedError



class TensorTransducer(Transducer):
	def __init__(self, T, R, s0, alphabet_in, alphabet_out, max_state=8):
		super().__init__(alphabet_in, alphabet_out, max_state)
		self.fsm = Params(T, R, s0)

		
	@staticmethod
	def run_fsm_with_values(inputs, R, T, s0):
		def f(s, x):
			y  = jnp.einsum('x,s,xsy->y', x, s, R)
			s1 = jnp.einsum('x,s,xst->t', x, s, T)
			return s1, (y, s1)

		_, (outputs, states) = jax.lax.scan(f, s0, inputs)
		#jax.debug.print("🤯 {outputs} 🤯", outputs=outputs)
		return outputs, jnp.vstack([s0, states])

	def __call__(self, inputs):
		inputs = prepare_str(inputs, self.alphabet_in_ext)
		return TensorTransducer.run_fsm_with_values(inputs, self.fsm.R, self.fsm.T, self.fsm.s0)

	def run_fsm(self, x):
		y, _ = self(x)
		return decode_str(y, self.alphabet_out)

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
		print(f"R = {self.fsm.R.shape}")
		print(f"Initial State = {self.fsm.s0.shape}")

	def get_edges_out(self, n):
		edges = []
		for i in range(self.fsm.R.shape[0]):
			for j in range(self.fsm.R.shape[2]):
				if self.fsm.R[i][n][j] > 0.1:
					edges.append((self.alphabet_in_ext[i], self.alphabet_out_ext[j]))
		return edges

	def iterate_state_io(self):
		for s1 in range(self.fsm.T.shape[1]):
			for s2 in range(self.fsm.T.shape[1]):
				for i in range(self.fsm.T.shape[0]):
					for o in range(self.fsm.T.shape[0]):
						yield (s1, s2, i, o)

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

	def to_nx_digraph(self):
		edges = {}
		### Mejorar los for anidados con un zip
		for s1, s2, i, o in self.iterate_state_io():
			if self.fsm.T[i][s1][s2] > 0.01 and self.fsm.R[i][s1][o] > 0.01:
				if (s1, s2) in edges:
					edges[(s1, s2)].append(f"\n{self.alphabet_in_ext[i]}/{self.alphabet_out_ext[o]}")
				else:
					edges[(s1, s2)] = [f"{self.alphabet_in_ext[i]}/{self.alphabet_out_ext[o]}"]

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

	def to_state_transducer(self):
		G, _, edges = self.to_nx_digraph()
		states = {n:i for i, n in enumerate(G.nodes)}
		
		edges_dict = {}
		for (s1, s2), xs in edges.items():
			if s1 in states and s2 in states:
				if not states[s1] in edges_dict:
					edges_dict[states[s1]] = []

				for x in xs:
					i, o = x.replace("\n", "").split("/")
					edges_dict[states[s1]].append((i, states[s2], o))

		initial_state = states[int(jnp.argmax(self.fsm.s0))]
		return StateTransducer(list(states.values()), edges_dict, initial_state, self.alphabet_in, self.alphabet_out)
		




class FunctionTransducer(Transducer):
	def __init__(self, f, alphabet_in, alphabet_out):
		self.f = f
		super().__init__(alphabet_in, alphabet_out, 8)	### Ver que hacer con ese max_states=8

	def __call__(self, inputs):
		return self.f(inputs), None

	def run_fsm(self, x):
		return self(x)
	
	def print(self):
		raise NotImplementedError
	
	def show(self, title="", verbose=0):
		raise NotImplementedError

	def to_state_transducer(self):
		raise NotImplementedError



class StateTransducer(Transducer):
	def __init__(self, states, edges, initial_state, alphabet_in, alphabet_out):
		super().__init__(alphabet_in, alphabet_out, len(states))
		self.states = states
		self.edges = edges
		self.initial_state = initial_state

	def get_edge(self, state, input_):
		for (i, s, o) in self.edges[state]:
			if input_ == i:
				return s, o
			
		return None, None

	def __call__(self, inputs):
		states = [self.initial_state]
		outputs = ""
		for i in inputs:
			state, output = self.get_edge(states[-1], i)	### [(i, s, o), ...]

			### Falta chequear if it None
			states.append(state)
			outputs += output
		
		return outputs, states

	def run_fsm(self, x):
		return self(x)

	def print(self):
		print(f"States: {self.states}")
		print(f"Edges: {self.edges}")
		print(f"Initial State: {self.initial_state}")

	def show(self, title="", verbose=0):
		G = nx.DiGraph()

		edges_dict = {}
		for s1, edges in self.edges.items():
			for i, s2, o in edges:
				print(i, o)
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

		return G, G.nodes

	def to_state_transducer(self):
		return self





