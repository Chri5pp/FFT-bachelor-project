import random
import time

from matplotlib import pyplot as plt
random.seed(1234)

# evaluatoin and interpolation has been optimized here unlike in the "fast subset convolution" file
# since my computer wasnt fast enough to run the benchmark otherwise.

# i havent written about how in my report to do this but it is very similar to the optimiztion
# done for the FFT base implementation.

#------------------------------------------------------------
#           MISC
#------------------------------------------------------------

def random_multilinear(n_args):
    return [random.randint(-9, 10) for _ in range(1 << n_args)]

#------------------------------------------------------------
#           Evaluation
#------------------------------------------------------------

def evaluate_multilinear_fast(P):
    # optimised for inplace non-recursive evaluation like the FFT base implementation
    n = len(P)
    length = 1

    while length < n:
        step = length << 1
        for i in range(0, n, step):
            for j in range(i, i + length):
                P[length + j] += P[j]
        length = step

    return P

#------------------------------------------------------------
#           Interpolation
#------------------------------------------------------------

def interpolate_multilinear_fast(points):
    # optimised for inplace non-recursive evaluation like the FFT base implementation
    n = len(points)
    length = 1

    while length < n:
        step = length << 1
        for i in range(0, n, step):
            for j in range(i, i + length):
                points[length + j] -= points[j]
        length = step

    return points

#------------------------------------------------------------
#           multiplication
#------------------------------------------------------------

def multiply_multilinear_fast(P, Q):
    evaluate_multilinear_fast(P)
    evaluate_multilinear_fast(Q)

    points = [P[i] * Q[i] for i in range(len(P))]

    return interpolate_multilinear_fast(points)

#------------------------------------------------------------
#           degree extraction
#------------------------------------------------------------

def naive_subset_convolution(P, Q):
    N = len(P)
    res = [0] * N

    for S in range(N):
        total = 0
        for T in range(N):
            if (T & ~S) == 0: #check if T is a subset of S
                total += P[T] * Q[S ^ T]
        res[S] = total

    return res

def slow_subset_convolution(P, Q):
    N = len(P)
    res = [0] * N

    for S in range(N):
        T = S # T loops over all subsets of S

        while True:
            res[S] += P[T] * Q[S ^ T]
            if T == 0:
                break
            T = (T - 1) & S # next subset of S

    return res


def fast_subset_convolution(P, Q):
    N = len(P)
    n = N.bit_length()

    res = [0] * N

    P_deg = [[0]*N for _ in range(n)]
    Q_deg = [[0]*N for _ in range(n)]

    for i in range(N):
        b = i.bit_count()
        P_deg[b][i] = P[i]
        Q_deg[b][i] = Q[i]

    for i in range(n):
        for j in range(n - i):
            conv = multiply_multilinear_fast(P_deg[i][:], Q_deg[j][:])

            for k in range(N):
                res[k] += conv[k] if k.bit_count() == i + j else 0

    return res


def benchmark():
    sizes = range(22)  # number of args for multinomials so 2^n terms
    naive_times = []
    slow_times  = []
    fast_times  = []

    def time_algorithm(func, a, b):
        start = time.perf_counter()
        func(a[:], b[:])   # copy to avoid mutation
        return time.perf_counter() - start

    for n in sizes:
        print(f"multiplying n={n}")

        a = random_multilinear(n)
        b = random_multilinear(n)
        
        if n <= 16: # naive is too slow for n > 15 so cant test that far
            res = time_algorithm(naive_subset_convolution, a, b)
            print(f"    Naive O(4^n) time:      {res:.4f} seconds")
            naive_times.append(res)

        res = time_algorithm(slow_subset_convolution, a, b)
        print(f"    Slow O(3^n) time:       {res:.4f} seconds")
        slow_times.append(res)

        res = time_algorithm(fast_subset_convolution, a, b)
        print(f"    Fast O(n^3 * 2^n) time: {res:.4f} seconds")
        fast_times.append(res)

        print(f"    Done with n={n}\n")

    plt.figure()
    plt.plot(sizes[:len(naive_times)], naive_times)
    plt.plot(sizes, slow_times)
    plt.plot(sizes, fast_times)

    plt.yscale("log")

    plt.xlabel("degree (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Polynomial multiplication comparison")
    plt.legend(["Naive O(4^n)", "Slow O(3^n)", "Fast O(n^3 * 2^n)"])
    plt.show()
    
benchmark()