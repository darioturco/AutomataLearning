import re


class Problem:
    def __init__(self, f, alphabet, xs, max_states=8):
        self.f = f
        self.alphabet = alphabet
        self.max_states = max_states
        self.xs = xs
        self.ys = ["".join([('1' if f(x[:i + 1]) else '0') for i in range(len(x))]) for x in xs]


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

def f11(x):
    return bool(re.fullmatch(r'(c|d)+(a|b)*', x))

def f12(x):
    return bool(re.fullmatch(r'', x))

def f13(x):
    return bool(re.fullmatch(r'', x))

def f14(x):
    return bool(re.fullmatch(r'', x))

def f15(x):
    return bool(re.fullmatch(r'', x))

def f16(x):
    return bool(re.fullmatch(r'', x))


problem1 = Problem(f1, ['a', 'b'], [], 2)
problem2 = Problem(f2, ['a', 'b'], ["aaabab", "aaaa"], 4)
problem3 = Problem(f3, ['a', 'b', 'c', 'd'], ["acbda", "abddacbdbd"], 6)
problem4 = Problem(f4, ['a', 'b'], [], 4)
problem5 = Problem(f5, ['a', 'b'], [], 10)
problem6 = Problem(f6, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ["7542025", "2025", "202587623"], 16)
problem7 = Problem(f7, ['a', 'b', 'c'], [], 12)
problem8 = Problem(f8, ['a', 'b', 'c', 'd'], ["daba", "aadaba"], 12)
problem9 = Problem(f9, ['a', 'b', 'c'], ["ccc", "ababccc", "bbbccc", "cbccc", "cccbaccc"], 8)
problem10 = Problem(f10, ['a', 'b', 'c'], [], 8)

problem11 = Problem(f11, ['a', 'b', 'c', 'd'], [], 8)
problem12 = Problem(f12, ['a', 'b', 'c'], [], 8)
problem13 = Problem(f13, ['a', 'b', 'c'], [], 8)
problem14 = Problem(f14, ['a', 'b', 'c'], [], 8)
problem15 = Problem(f15, ['a', 'b', 'c'], [], 8)
problem16 = Problem(f16, ['a', 'b', 'c'], [], 8)

