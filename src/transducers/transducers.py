from collections import namedtuple
import graphviz
import networkx as nx
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
			y, s = self.run_fsm([x])
			y0 = jnp.array([prepare_str(y0, self.alphabet_out_ext)])
			error += jnp.square(y-y0).sum()
			#error += 0.0 if s is None else entropy(s.mean(0)) * entropy_weight
		return error

	def cancat_error_square(self, xs, ys0):
		x = self.separate_char.join(xs) + self.separate_char
		y0 = self.separate_char.join(ys0) + self.separate_char
		x = prepare_str(x, self.alphabet_in_ext)
		y0 = prepare_str(y0, self.alphabet_out_ext)
		fsm = self.fsm
		y, s = TensorTransducer.run_fsm_with_values(x, fsm.R, fsm.T, fsm.s0)
		return jnp.square(y - y0).sum()

	def __call__(self, x):
		raise NotImplementedError

	def run_fsm(self, inputs):
		raise NotImplementedError

	def to_state_transducer(self):
		raise NotImplementedError



class TensorTransducer(Transducer):
	def __init__(self, T, R, s0, alphabet_in, alphabet_out, max_state=8):
		super().__init__(alphabet_in, alphabet_out, max_state)
		self.fsm = Params(T, R, s0)

	@staticmethod
	def run_fsm_with_values(inputs, R, T, s0):
		# x.shape = []
		# R.shape = []
		# T.shape = []
		def f(s, x):
			s1 = jnp.einsum('ix,is,xst->it', x, s, T)
			y  = jnp.einsum('ix,is,xsy->iy', x, s, R)

			return s1, (y, s1)

		init_states = jnp.array([s0 for _ in range(inputs.shape[0])])
		_, (outputs, states) = jax.lax.scan(f, init_states, inputs.transpose((1,0,2)))
		return outputs.transpose((1,0,2)), jnp.vstack([[init_states], states]).transpose((1,0,2))

	""" Returns the string corresponding to x """
	def __call__(self, x):
		y, _ = self.run_fsm([x])
		return decode_str(y, self.alphabet_out_ext)

	def run_fsm(self, inputs):
		inputs = jnp.array([prepare_str(x, self.alphabet_in_ext) for x in inputs])
		return TensorTransducer.run_fsm_with_values(inputs, self.fsm.R, self.fsm.T, self.fsm.s0)

	def __repr__(self):
		return f"""
R = {self.fsm.R.shape}
T = {self.fsm.T.shape}
Initial State = {self.fsm.s0.shape}"""

	def print(self):
		print(self.__repr__())

	def show(self, path=None, title="", node_size=500, verbose=0):
		state_transducer = self.to_state_transducer()
		return state_transducer.show(path=path, title=title, node_size=node_size, verbose=verbose)

	def get_edges_out(self, n):
		edges = []
		for i in range(self.fsm.R.shape[0]):
			for j in range(self.fsm.R.shape[2]):
				if self.fsm.R[i][n][j] > 0.1:
					edges.append((self.alphabet_in_ext[i], self.alphabet_out_ext[j]))
		return edges

	### Completar
	def to_nx_digraph(self):
		edges_dict = {}	# [(s1, s2): ["i1/o1", "\ni2/o2"...]]
		for s_ in range(self.max_state):
			for i, c in enumerate(self.alphabet_in):
				new_s = int(jnp.einsum('x,s,xst->t', jnp.eye(1, self.char_n_in, i)[0], jnp.eye(1, self.max_state, s_)[0], self.fsm.T).argmax())
				output_index = int(jnp.einsum('x,s,xsy->y', jnp.eye(1, self.char_n_in, i)[0], jnp.eye(1, self.max_state, s_)[0],self.fsm.R).argmax())
				o = self.alphabet_out[output_index]
				if (s_, new_s) in edges_dict:
					edges_dict[(s_, new_s)].append(f"\n{c}/{o}")
				else:
					edges_dict[(s_, new_s)] = [f"{c}/{o}"]

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

	### Revisar
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
	def __init__(self, f, alphabet_in, alphabet_out, max_states=8):
		self.f = f			# The function return the string corresponding at the input
		super().__init__(alphabet_in, alphabet_out, max_states)

	def __repr__(self):
		return self.__repr__()

	def __call__(self, x):
		return self.f(x)

	def run_fsm(self, inputs):
		outputs = [self(x) for x in inputs]
		return jnp.array([prepare_str("".join(y), self.alphabet_out_ext) for y in outputs]), None

	def print(self):
		print(self.__repr__())
	
	def show(self, path=None, title="", node_size=500, verbose=0):
		raise NotImplementedError

	def to_state_transducer(self):
		raise NotImplementedError

class StateTransducer(Transducer):
	""" Generate a DFA with n states and nt transition by each state, the transitions are random
			and probability end_p that a state is a final state """
	### Revisar y completar
	@classmethod
	def generate_random_dfa(cls, alphabet_in, alphabet_out, n, nt):
		pass
		#states = [i for i in range(n)]

		#t_per_state = min(nt, len(states), len(alphabet))
		#edges_dict = {
		#	s1: [(a, s2) for a, s2 in zip(random.sample(alphabet, k=t_per_state), random.sample(states, k=t_per_state))]
		#	for s1 in states}



		#return cls(states, edges_dict, 0, alphabet_in, alphabet_out)

	def __init__(self, states, edges, initial_state, alphabet_in, alphabet_out):
		super().__init__(alphabet_in, alphabet_out, len(states)+1)
		self.states = list(set(states))
		self.edges = edges					# {s1: [(i, s_i, o), ...], s2:...}
		self.initial_state = initial_state
		self.trap_state = -1
		self.actual_state = self.initial_state

	def next_state(self, state, input_):
		if state not in self.edges:
			return None
		if state == self.trap_state:
			return self.trap_state

		for (i, s, o) in self.edges[state]:
			if input_ == i:
				return s, o
			
		return None, None

	### Completar
	def __call__(self, x):
		pass

	def run_fsm(self, inputs):
		res = []
		states_res = []

		for x in inputs:
			states = [self.initial_state]
			outputs = []

			for i in x:
				state, output = self.next_state(states[-1], i)  # [(i, s, o), ...]
				if state is None:
					### Falta ver que se hace con el output
					state = self.trap_state

				states.append(state)
				outputs += output

			res.append(prepare_str(outputs, self.alphabet_out_ext))
			states_res.append(states.copy())

		return jnp.array(res), states_res

	def __repr__(self):
		return f"""
States: {self.states}
Edges: {self.edges}
Initial State: {self.initial_state}"""

	def print(self):
		print(self)

	def show(self, view=True, name="Transducer", verbose=0):
		if verbose:
			self.print()

		graph = graphviz.Digraph(name)
		for i, s in enumerate(self.states):
			shape = 'doublecircle' if s == self.initial_state else 'circle'
			graph.node(str(s), shape=shape, color='black',  fillcolor="white", style='filled')

		for s1, xs in self.edges.items():
			for c, s2, o in xs:
				graph.edge(str(s1), str(s2), label=f"{c}/{o}")

		if view:
			graph.render(directory='graphs', view=True)
		return graph
		
	def to_state_transducer(self):
		return self

	def add_state(self, state):
		self.states = list(set(self.states + [state]))
		if state not in self.edges:
			self.edges[state] = []

	def add_transition(self, s1, c, s2, o):
		# assert s1 in self.states and s2 in self.states and c in self.alphabet ### Revisar

		if s1 not in self.edges:
			self.add_state(s1)

		### Check that the edges is not present yet
		self.edges[s1].append((c, s2, o))

	def step(self, x):
		new_state, output = self.next_state(self.actual_state, x)
		if new_state is None:
			new_state = self.trap_state
			### Ver que hacer con el output
		self.actual_state = new_state
		return new_state, output

	def reset(self):
		self.actual_state = self.initial_state