import numpy as np
from random import uniform
from time import time
import math, os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import odr


def get_plot_data(func, n, window_size, degree):
    result = []
    for dim in range(1, 4):
        max_x = int(math.floor(math.pow(n+1, 1/dim)))
        step = max(1, max_x//10)
        x = [i**dim for i in range(1,max_x+1, step)]
        #pool = Pool(os.cpu_count())
        #ttimes = pool.starmap(get_times, ((func,window_size, int(math.pow(i,1/dim)), dim) for i in x))
        ttimes = [get_times(func,window_size, int(math.pow(i,1/dim)), dim) for i in x]
        y = []
        err = []
        for time, time_var in ttimes:
            y.append(time)
            err.append(time_var)
      
        x_app = np.linspace(1, n+1, num = 2*n)
        y_pred = polynomial_approx(degree, x, y, x_app)
        result.append((dim,x, y, err, x_app, y_pred))
    return result

def polynomial_approx(degree, x, y, x_new):
    polynomial_function = lambda B, x: np.poly1d(B)(x)
    model = odr.Model(polynomial_function)
    data_to_fit = odr.Data(x, y)
    job = odr.ODR(data_to_fit, model, beta0= [uniform(0, 1) for _ in range(degree+1)])
    results = job.run()
    return polynomial_function(results.beta, x_new)

def get_times(func, window_size, *args):
    total_time = 0
    deltas = []
    for _ in range(window_size):
        delta = func(*args)
        total_time+=delta
        deltas.append(delta)
    total_time /= window_size
    return (total_time,(max(deltas) - min(deltas))/2)
    
def multiply_dim_lists(a, b, dim):
    if dim == 1: return list(map(lambda x,y: x*y, a, b)) 
    return [multiply_dim_lists(a[i], b[i],dim-1) for i in range(len(a))]

def multiply_lists_time(len, dim):
    a = [uniform(0,1) for _ in range(len)]
    b = [uniform(0,1) for _ in range(len)]
    for __ in range(1, dim):
        a = [a for _ in range(len)]
        b = [b for _ in range(len)]
    start_time = time()
    multiply_dim_lists(a, b, dim)
    return time()-start_time
    
def multiply_nparrays_time(len, dim):
    a = np.random.random([len for _ in range(dim)])
    b = np.random.random([len for _ in range(dim)])
    start_time = time()
    a*b
    return time()-start_time

def do_plots(n, window_size, degree):
    colors = ["red", "green", "blue"]
    titles = {}
    titles["Lists"] = multiply_lists_time
    titles["Numpy arrays"] = multiply_nparrays_time
    
    cel_num = 1
    for title, func in titles.items():
        plt.subplot(len(titles), 1, cel_num)
        plt.title(title)
        plt.xlabel("$N$")
        plt.ylabel("$T$")
        for dim, x, y, err, x_app, y_app in get_plot_data(func, n, window_size, degree):
            color  = colors[dim-1]
            plt.errorbar(x, y, err, fmt = color[0]+"o")
            plt.plot(x_app, y_app, '--', color = color)
        cel_num+=1
        plt.legend([f'{i} dimensions' for i in range(1, 4)])
    
    plt.show()

do_plots(10000, 100, 3)