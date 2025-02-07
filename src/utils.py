import jax.numpy as jnp
from jax import nn
import random

full_alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ #.,*()[]{}@!$^&-+?_=|`~<>"
lambda_char = 'λ'

def f_char2int(alphabet):
  return lambda c: alphabet.index(c)

def hardmax(x):
  return nn.one_hot(x.argmax(-1), x.shape[-1])

def decode_fsm(params, hard=False):
  T, R, s0 = params
  f = hardmax if hard else nn.softmax
  return f(T), f(R), f(s0)

def prepare_str(s, alphabet):
  return nn.one_hot(list(map(f_char2int(alphabet), s)), len(alphabet))

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
  xss = []
  yss = []
  for _ in range(l):
    xs = probabilistic_sample(alphabet, p)
    ys = ""
    for i in range(len(xs)):
      ys += '1' if f(xs[:i+1]) else '0'
    xss.append(xs)
    yss.append(ys)

  return xss, yss

def cartesian_product(list1, list2):
  return [(i, j) for i in list1 for j in list2]

