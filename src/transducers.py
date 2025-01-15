from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from src.utils import decode_fsm, entropy, prepare_str, FSM, Params, Stats, TrainState, TrainResult

class TensorTransducer:
  def __init__(self, T, R, s0, alphabet, max_state=8):
    self.fsm = decode_fsm(Params(T, R, s0), hard=True)
    self.alphabet = alphabet
    self.alphabet_ext = alphabet + ["2"]
    self.CHAR_N = len(alphabet) + 1
    self.STATE_MAX = max_state
    self.char_dict = {0: '0', 1: '1', 2: '2'} ### Cambiar

  @staticmethod
  def run_fsm_with_values(inputs, R, T, s0):
    def f(s, x):
      y  = jnp.einsum('x,s,xsy->y', x, s, R)
      s1 = jnp.einsum('x,s,xst->t', x, s, T)
      return s1, (y, s1)

    _, (outputs, states) = jax.lax.scan(f, s0, inputs)
    return outputs, jnp.vstack([s0, states])

  def __call__(self, inputs):
    inputs = prepare_str(inputs, self.CHAR_N)
    return TensorTransducer.run_fsm_with_values(inputs, self.fsm.R, self.fsm.T, self.fsm.s0)

  def run_fsm(self, x):
    y, _ = self(x)
    index = y.argmax(axis=1)
    return "".join([self.char_dict[i] for i in index.tolist()][:-1] + ['2'])
  
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
    print(f"T = {self.fsm.T}")
    print(f"R = {self.fsm.R}")
    print(f"Initial State = {self.fsm.s0}")

  def get_edges_out(self, n):
    edges = []
    for i in range(self.fsm.R.shape[0]):
      for j in range(self.fsm.R.shape[2]):
        if self.fsm.R[i][n][j] > 0.1:
          edges.append((self.alphabet_ext[i], self.alphabet_ext[j]))
    return edges

  def show(self, title="", verbose=0):
    if verbose:
      self.print()
    
    edges = {}
    for s1 in range(self.fsm.T.shape[1]):
      for s2 in range(self.fsm.T.shape[1]):
        for i in range(self.fsm.T.shape[0]):
          for o in range(self.fsm.T.shape[0]):
            if self.fsm.T[i][s1][s2] > 0.01 and self.fsm.R[i][s1][o] > 0.01:
              if (s1, s2) in edges:
                edges[(s1, s2)].append(f"\n{self.alphabet_ext[i]}/{self.alphabet_ext[o]}")
              else:
                edges[(s1, s2)] = [f"{self.alphabet_ext[i]}/{self.alphabet_ext[o]}"]

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

    pos = nx.circular_layout(G)
    color_map = ["green" if n == initial_state else "lightblue" for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, arrowsize=16, font_size=8)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show the plot
    plt.title(title)
    plt.show()



    



class FunctionTransducer:
  def __init__(self, f, alphabet):
    self.f = f
    self.alphabet = alphabet
    self.STATE_MAX = 8
    self.CHAR_N = len(alphabet) + 1

  def __call__(self, inputs):
    return self.f(inputs)

  def run_fsm(self, x):
    return self(x)