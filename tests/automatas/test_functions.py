import re

class Problem:
    def __init__(self, f, alphabet, xs, ys, max_states=8):
        self.f = f
        self.alphabet = alphabet
        self.max_states = max_states
        self.xs = xs
        self.ys = ys

# Accepts only string that end with 'a'
def f1(x):
    return x[-1] == 'a'

# Return if the substring "aaa" is in the input string
def f2(x):
    return "aaa" in x

# Return if the substring "acbd" is not in the input string
def f3(x):
    return not "acbd" in x

# Return if the amount of 'b' is an even number
def f4(x):
    return x.count('b') % 2 == 0


# Return if the string start with a and repeats 0 of more times the string "ab"
def f5(x):
    return bool(re.fullmatch(r'a(ab)*', x))

# Return if the string is equal to "2025"
def f6(x):
    return x == "2025"

def f7(x):
    return bool(re.fullmatch(r'(ca)*b+', x))

def f8(x):
    return bool(re.fullmatch(r'(aaa)*dab', x))

def f9(x):
    return bool(re.fullmatch(r'(a|b)ccc', x))

def f10(x):
    return bool(re.fullmatch(r'c(a|b)+', x))

problem1 = Problem(f1, ['a', 'b'], [], [], 2)
problem2 = Problem(f2, ['a', 'b'],  ["aaabab", "aaaa"], ["001111", "0011"], 4)
problem3 = Problem(f3, ['a', 'b', 'c', 'd'],  ["acbda", "abddacbdbd"], ["11100", "1111111000"], 6)
problem4 = Problem(f4, ['a', 'b'],  [], [], 4)
problem5 = Problem(f5, ['a', 'b'],  [], [], 10)
problem6 = Problem(f6, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],  ["7542025", "2025", "202587623"], ["0000000", "0001", "000100000"], 16)
problem7 = Problem(f7, ['a', 'b', 'c'],  [], [], 12)
problem8 = Problem(f8, ['a', 'b', 'c', 'd'],  [], [], 12)
problem9 = Problem(f9, ['a', 'b', 'c'],  [], [], 10)
problem10 = Problem(f10, ['a', 'b', 'c'],  [], [], 8)
