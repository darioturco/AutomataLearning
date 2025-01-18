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
    