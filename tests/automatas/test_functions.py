import re


class Problem:
    def __init__(self, f, alphabet, xs, max_states=8):
        self.f = f
        self.alphabet = alphabet
        self.max_states = max_states
        self.xs = xs
        self.ys = ["".join([('1' if f(x[:i + 1]) else '0') for i in range(len(x))]) for x in xs]
        self.description = f.__doc__
        self.num = int(f.__doc__[:f.__doc__.index(":")])

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
problem12 = Problem(f12, ['a', 'b'], ["aaa", "abaabaaa"], 8)
problem13 = Problem(f13, ['a', 'b'], ["aaaa", "aaaabaaaa", "abaaa", "aaaaaaa"], 8)
problem14 = Problem(f14, ['a'], ["aa", "aaaaaaaa"], 8)
problem15 = Problem(f15, ['a', 'b'], ["aa", "bb", "abaa", "babb", "abababab"], 8)
problem16 = Problem(f16, ['a', 'b'], ["aa", "bb", "abaa", "babb", "abababab"], 8)
