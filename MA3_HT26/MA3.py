""" MA3.py

Student: Jacob Ersson
Mail: jacob.ersson05@gmail.com
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from functools import reduce
from numba import njit
import time
import concurrent.futures as future

# Exc1
def approximate_pi(n):
    nc = []
    nk = []

    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            nc.append([x, y])
        else:
            nk.append([x, y])

    approx_pi = 4 * len(nc) / n

    plt.scatter([x for x, y in nc], [y for x, y in nc], color="r", label="n_c", s=5)
    plt.scatter([x for x, y in nk], [y for x, y in nk], color="b", label="n_k", s=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.5)
    #plt.savefig("circle100000.png", bbox_inches="tight")
    #plt.show()

    return approx_pi



# Exc2, approximation
def sphere_volume(n, d): 
    vs = [[random.uniform(-1,1) for ii in range(1, d+1)] for jj in range(n)]

    def inside_sphere(v):
        if reduce(lambda x,y : x + y**2, v, 0) <= 1:
            return True
        return False

    nc = list(filter(inside_sphere, vs))

    return len(nc) * 2**d / n

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points

    # d is the number of dimensions of the sphere 
    V = m.pi**(d/2) / m.gamma((d/2 + 1))
    return V


#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    vs = [[random.uniform(-1,1) for ii in range(1, d+1)] for jj in range(n)]

    def inside_sphere(v):
        if reduce(lambda x,y : x + y**2, v, 0) <= 1:
            return True
        return False

    nc = list(filter(inside_sphere, vs))

    return len(nc) * 2**d / n
    



#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel2(n, d, vs, np=10):
    def inside_sphere(v):
        if reduce(lambda x,y : x + y**2, v, 0) <= 1:
            return True
        return False

    nc = list(filter(inside_sphere, vs))

    return len(nc) * 2**d / n




def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Exc2: Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print()

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Exc2: Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print()

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    #sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}s")
        # run1: 5.883s
        # run2: 6.032s.   ### No Numba
        # run3: 5.945s

        # run1: 1.616s
        # run2: 1.614s.   ### Numba
        # run3: 1.624s

    start = pc()
    #sphere_volume_numba(n, d)
    stop = pc()

    print(f"What is numba time?: {stop-start}s")
    print()

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
        # 4.53 s
        # 4.58 s
        # 4.52 s


    vs = [[random.uniform(-1,1) for ii in range(1, d+1)] for jj in range(n)]
    chunk = len(vs) // 10

    processes = [vs[i:i + chunk] for i in range(0, len(vs), chunk)]

    start=pc()
    with future.ProcessPoolExecutor(10) as ex:

        futures = [ex.submit(sphere_volume_parallel2, n, d, process) for process in processes]
        results = [f.result() for f in futures]

        total = sum(results)

    stop=pc()
    print(f"Exc4: Paralell time of {d} and {n}: {stop-start}")
        # 2.88 s
        # 2.96 s
        # 2.91 s


if __name__ == '__main__':
	main()
