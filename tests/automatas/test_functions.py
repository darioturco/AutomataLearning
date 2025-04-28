import re
from src.automatas.automatas import FunctionAutomata

class Problem:
    def __init__(self, automata, xs, description, num):
        self.f = automata.run_fsm
        self.target_automata = automata
        self.alphabet = automata.alphabet
        self.max_states = automata.max_state
        self.xs = xs
        self.ys = ["".join([('1' if self.f(x[:i + 1]) else '0') for i in range(len(x))]) for x in xs]
        self.description = description
        self.num = num

def f1(x):
    """ 1: Accepts only string that end with 'a'"""
    return x[-1] == 'a'

def f2(x):
    """ 2: Return if the substring "aaa" is in the input string"""
    return "aaa" in x

def f3(x):
    """ 3: Return if the substring "acbd" is not in the input string"""
    return not "acbd" in x

def f4(x):
    """ 4: Return if the amount of 'b' is an even number"""
    return x.count('b') % 2 == 0

def f5(x):
    """ 5: Return if the string start with a and repeats 0 of more times the string 'ab'"""
    return bool(re.fullmatch(r'a(ab)*', x))

def f6(x):
    """ 6: Return if the string is equal to "2025" """
    return x == "2025"

def f7(x):
    """ 7: Full match with regex (ca)*b+ """
    return bool(re.fullmatch(r'(ca)*b+', x))

def f8(x):
    """ 8: Full match with regex (aaa)*dab """
    return bool(re.fullmatch(r'(aaa)*dab', x))

def f9(x):
    """ 9: Full match with regex (a|b)ccc """
    return bool(re.fullmatch(r'(a|b)ccc', x))

def f10(x):
    """ 10: Full match with regex c(a|b)+ """
    return bool(re.fullmatch(r'c(a|b)+', x))

def f11(x):
    """ 11: Full match with regex (c|d)+(a|b)* """
    return bool(re.fullmatch(r'(c|d)+(a|b)*', x))

def f12(x):
    """ 12: Termina en 'aaa' """
    return bool(re.fullmatch(r'(a|b)*aaa', x))

def f13(x):
    """ 13: Solo acepta la cadenas que tengan solo 'a' """
    return bool(re.fullmatch(r'a+', x))

def f14(x):
    """ 14: Acepta todas las cadenas salvo "aa" """
    return not bool(re.fullmatch(r'aa', x))

def f15(x):
    """ 15: Tener la subcadena 'aa' o la cadena 'bb' """
    return ("aa" in x) or ("bb" in x)

def f16(x):
    """ 16: Tener la subcadena 'aa' y la cadena 'bb' """
    return ("aa" in x) and ("bb" in x)


def get_doc_num(f): ### Pasar a utils
    return int(f.__doc__[:f.__doc__.index(":")])


problem1 = Problem(FunctionAutomata(f1, ['a', 'b']), [], f1.__doc__, get_doc_num(f1))
problem2 = Problem(FunctionAutomata(f2, ['a', 'b']), ["aaabab", "aaaa"], f2.__doc__, get_doc_num(f2))
problem3 = Problem(FunctionAutomata(f3, ['a', 'b', 'c', 'd']), ["acbda", "abddacbdbd"], f3.__doc__, get_doc_num(f3))
problem4 = Problem(FunctionAutomata(f4, ['a', 'b']), [], f4.__doc__, get_doc_num(f4))
problem5 = Problem(FunctionAutomata(f5, ['a', 'b']), [], f5.__doc__, get_doc_num(f5))
problem6 = Problem(FunctionAutomata(f6, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']), ["7542025", "2025", "202587623"], f6.__doc__, get_doc_num(f6))
problem7 = Problem(FunctionAutomata(f7, ['a', 'b', 'c']), [], f7.__doc__, get_doc_num(f7))
problem8 = Problem(FunctionAutomata(f8, ['a', 'b', 'c', 'd']), ["daba", "aadaba"], f8.__doc__, get_doc_num(f8))
problem9 = Problem(FunctionAutomata(f9, ['a', 'b', 'c']), ["ccc", "ababccc", "bbbccc", "cbccc", "cccbaccc"], f9.__doc__, get_doc_num(f9))
problem10 = Problem(FunctionAutomata(f10, ['a', 'b', 'c']), [], f10.__doc__, get_doc_num(f10))
problem11 = Problem(FunctionAutomata(f11, ['a', 'b', 'c', 'd']), [], f11.__doc__, get_doc_num(f11))
problem12 = Problem(FunctionAutomata(f12, ['a', 'b']), ["aaa", "abaabaaa"], f12.__doc__, get_doc_num(f12))
problem13 = Problem(FunctionAutomata(f13, ['a', 'b']), ["aaaa", "aaaabaaaa", "abaaa", "aaaaaaa"], f13.__doc__, get_doc_num(f13))
problem14 = Problem(FunctionAutomata(f14, ['a']), ["aa", "aaaaaaaa"], f14.__doc__, get_doc_num(f14))
problem15 = Problem(FunctionAutomata(f15, ['a', 'b']), ["aa", "bb", "abaa", "babb", "abababab"], f15.__doc__, get_doc_num(f15))
problem16 = Problem(FunctionAutomata(f16, ['a', 'b']), ["aa", "bb", "abaa", "babb", "abababab"], f16.__doc__, get_doc_num(f16))

