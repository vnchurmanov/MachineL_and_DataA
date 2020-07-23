from matplotlib import pylab as plt
import numpy as np
import math
from scipy.optimize import minimize, differential_evolution


# Calculate f(x) if x is known (num)
def f_x(num):
    return math.sin(num / 5.) * math.exp(num / 10.) + 5 * math.exp(-num / 2.)


"""Task 1. Minimization of smooth function"""

# 1 Make a graph with matplotlib

x = np.arange(1, 30, 0.01)
f = np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
plt.plot(x, f)

# 5 Trying to get minimum with standard func scipy minimize (Nelder-Mead)
x1 = np.array([18])
x_min = minimize(f_x, x1, method='nelder-mead')
print('Nelder-Mead: ', x_min.x)

# 6 Trying to get minimum with scipy minimize (BFGS), x0 = 2
x2 = np.array([2])
x_min_BFGS_1 = minimize(f_x, x2, method='BFGS')
print('BFGS: (x0 = 2)', x_min_BFGS_1['fun'])

# 7 Trying to get minimum with scipy minimize (BFGS), x0 = 30
x3 = np.array([30])
x_min_BFGS_2 = minimize(f_x, x3, method='BFGS')
print('BFGS: (x0 = 30)', x_min_BFGS_2['fun'])

"""Task 2"""

# 4 Get min with method of differential evolution, bounds from 1 to 30

bounds = [(1, 30)]
result = differential_evolution(f_x, bounds)
print("Differential evolution: (bounds = 1, 30)", result['fun'])

"""Task 3. Minimization of not smooth function"""


# Calculate h(x) = int(f(x)) if x is known (num)
def h_x(num):
    return int(math.sin(num / 5.) * math.exp(num / 10.) + 5 * math.exp(-num / 2.))


# 2 Make a graph with matplotlib

x = np.arange(1, 30, 0.01)
h = np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
# 'Cause numpy array don't want to be int, write 'for cycle' with result of list of int values
int_func_of_f = []
for num in h:
    int_func_of_f.append(int(num))
plt.plot(x, int_func_of_f)
plt.show()

# 3 Trying to get minimum with scipy minimize (BFGS), x0 = 30
x = np.array([30])
res = minimize(h_x, x, method='BFGS')
print("Non smooth BFGS: (x0 = 30)", res["fun"])

# 4 Get min with method of differential evolution, bounds from 1 to 30

bounds = [(1, 30)]
result = differential_evolution(h_x, bounds)
print("Differential evolution of non smooth func: (bounds = 1, 30)", result["fun"])
