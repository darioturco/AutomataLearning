from collections import namedtuple
import jax.numpy as jnp
from jax import nn

FSM = namedtuple('FSM', 'T R s0')
Params = namedtuple('Params', 'T R s0')
Stats = namedtuple('Stats', 'total error entropy states_used')
TrainState = namedtuple('TrainState', 'params opt_state')
TrainResult = namedtuple('TrainResult', 'params eval logs')

def hardmax(x):
  return nn.one_hot(x.argmax(-1), x.shape[-1])

def decode_fsm(params: Params, hard=False) -> FSM:
  T, R, s0 = params
  f = hardmax if hard else nn.softmax
  return FSM(f(T), f(R), f(s0))

def prepare_str(s, char_n):
  return nn.one_hot(list(map(int, s)), char_n)

def entropy(p):
  return jnp.where(p>0.0, -p*jnp.log(p), 0.0).sum()