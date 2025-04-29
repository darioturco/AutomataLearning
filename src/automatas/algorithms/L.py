from src.automatas.automatas import StateAutomata
from src.utils import lambda_char, probabilistic_sample
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.base import SUL


class WrapperSUL(SUL):
    def __init__(self, target_dfa):
        super().__init__()
        self.target_dfa = target_dfa

    def step(self, letter):
        ### Ningun tipo de automata tiene implementada la funcion step
        ### Crear un automata de un nuevo tipo o hacer que StateAutomata guarde el estado actual y tenga step (luego convertir cualquier tipo de automata a SatateAutomata)
        return self.target_dfa.step(letter)
    
    def reset(self):
        ### Implementar funcion de reset en nuestra implemntacion de StateAutomata
        return self.target_dfa.reset()

# Algorthm from "Learning regular sets from queries and counterexamples"
class LStartLearner:
    def __init__(self, alphabet, target_system, verbose=0):
        self.alphabet = alphabet
        self.sul = WrapperSUL(target_system)
        self.eq_oracle = RandomWMethodEqOracle(alphabet, self.sul)
        self.verbose = verbose

    def learn(self):
        model = run_Lstar(self.alphabet, self.sul, self.eq_oracle, 'dfa', cache_and_non_det_check=False)
        ### Ver que clase es "model" y pasarla a nuestra clase de StateAutomata
        return model