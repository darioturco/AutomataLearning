from collections import namedtuple
import jax.numpy as jnp
from jax import nn
import random

FSM = namedtuple('FSM', 'T R s0')
Params = namedtuple('Params', 'T R s0')
Stats = namedtuple('Stats', 'total error entropy states_used')
TrainState = namedtuple('TrainState', 'params opt_state')
TrainResult = namedtuple('TrainResult', 'params eval logs')

full_alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "

def f_char2int(alphabet):
  return lambda c: alphabet.index(c)

def hardmax(x):
  return nn.one_hot(x.argmax(-1), x.shape[-1])

def decode_fsm(params: Params, hard=False) -> FSM:
  T, R, s0 = params
  f = hardmax if hard else nn.softmax
  return FSM(f(T), f(R), f(s0))

def prepare_str(s, alphabet):
  return nn.one_hot(list(map(f_char2int(alphabet), s)), len(alphabet))

def entropy(p):
  return jnp.where(p>0.0, -p*jnp.log(p), 0.0).sum()

def get_separate_char(alphabet):
  return [c for c in full_alphabet if c not in alphabet][0]

def generate_data_set(f, alphabet, records=16, min_length=1, max_length=8):
  res = {}
  while len(res) < records:
    x = ''.join(random.choices(alphabet, k=random.randint(min_length, max_length)))
    res[x] = f(x)

  return list(res.keys()), list(res.values())