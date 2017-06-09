from scipy.optimize import fmin
def rosen(x):  # The Rosenbrock function
    #print sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

xopt = fmin(rosen, x0)