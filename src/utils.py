import jax.numpy as jnp
from jax import nn
import random
import pickle

full_alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ #.,*()[]{}@!$^&-+?_=|`~<>"
lambda_char = 'λ'

def f_char2int(alphabet):
  return lambda c: alphabet.index(c)

def hardmax(x):
  return nn.one_hot(x.argmax(-1), x.shape[-1])

def decode_fsm(params, hard=False):
  T, A, s0 = params
  f = hardmax if hard else nn.softmax
  return f(T), f(A), f(s0)

def prepare_str(s, alphabet, padding=0):
  return nn.one_hot(list(map(f_char2int(alphabet), s + (alphabet[-1]*padding))), len(alphabet))

def decode_str(x, alphabet):
  return "".join([alphabet[i] for i in x.argmax(axis=1).tolist()])

def entropy(p):
  return jnp.where(p>0.0, -p*jnp.log(p), 0.0).sum()

def get_separate_char(alphabet):
  return [c for c in full_alphabet if c not in alphabet][0]





def generate_data_set(f, alphabet_in, records=16, min_length=1, max_length=8, verbose=0):
  res = {}
  while len(res) < records:
    x = ''.join(random.choices(alphabet_in, k=random.randint(min_length, max_length)))
    res[x] = f(x)

  xs, ys = list(res.keys()), list(res.values())
  if verbose:
    print(xs, ys)

  return xs, ys

def probabilistic_sample(alphabet, p=0.1, can_be_empty=False):
  res = "" if can_be_empty else random.choice(alphabet)
  while random.random() < p:  # Flip a coin with probability 'mu'
    res += random.choice(alphabet)
  return res

def sample_dataset(f, alphabet, p=0.1, l=8):
  if f is None:
    f = lambda x: 0
  xss = set()
  i = 0
  while len(xss) < l and i < l*1000:  # Emergency exit when 1000 repeated string occurs
    xss.add(probabilistic_sample(alphabet, p))
    i = i+1

  yss = ["".join(['1' if f(xs[:i+1]) else '0' for i in range(len(xs))]) for xs in xss]

  return list(xss), yss

### Completar
def probability_of_word(l, mu, alphabet):
  return mu * ((1 - mu) / len(alphabet)) ** len(l)

def language_distance(l1, l2, mu, alphabet):
  l = set(l1).symmetric_difference(set(l2))
  return sum([probability_of_word(s, mu, alphabet) for s in l])

### Completar
def estimate_language_distance(automata1, automata2, mu, alpha, gamma, alphabet):
  max_l = round(jnp.exp(2/gamma) / (2 * alpha ** 2)) + 1
  xs, _ = sample_dataset(None, alphabet, p=mu, l=max_l)
  l1, l2 = [], []
  for s in xs:
    if automata1.run_fsm(s):
      l1.append(s)
    if automata2.run_fsm(s):
      l2.append(s)

def cartesian_product(list1, list2):
  return [(i, j) for i in list1 for j in list2]

def save_pickle(data, path):
  with open(path, "wb") as f:
    pickle.dump(data, f)

def load_pickle(path):
  with open(path, "rb") as f:
    loaded_variable = pickle.load(f)
  return loaded_variable

