
def derive(f, x, h=0.0001):
    # TODO: implement this function
    derivate = (f(x+h)-f(x))/h
    return derivate