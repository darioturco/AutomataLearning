class Problem:
    def __init__(self, f, alphabet_in, alphabet_out):
        self.f = f
        self.alphabet_in = alphabet_in
        self.alphabet_out = alphabet_out

# Replace '1' by '0' and '0' by '1'
def f1(x):
    d = {'0': '1', '1': '0'}
    return "".join([d[c] for c in x])

# Take the first caharacter [0-9] and emits that character
def f2(x):
    if x == "":
        return ""
    
    c = x[0]
    return c * len(x)


problem1 = Problem(f1, ['0', '1'], ['0', '1'])
problem2 = Problem(f2, ['0', '1', '2', '3'], ['0', '1', '2', '3'])
#problem3 = Problem(f2, ['0', '1', '2', '3'], ['0', '1', '2', '3'])