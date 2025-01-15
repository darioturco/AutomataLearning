def f1(x):
    d = {'0': '1', '1': '0', '2':'2'}
    return "".join([d[c] for c in x])