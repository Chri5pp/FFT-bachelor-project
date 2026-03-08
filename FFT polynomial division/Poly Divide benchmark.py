from cmath import exp, pi
import random
import time
import matplotlib.pyplot as plt


def remove_trailing_zeros(A, epsilon=1e-10):
    while len(A) > 1 and abs(A[-1]) < epsilon:
        A.pop()
    return A

def FFT(x):
    N = len(x)
    if N <= 1: return x
    
    even = FFT(x[0::2])
    odd = FFT(x[1::2])

    w_n = exp(-2j * pi / N)
    w = 1 + 0j
    y = [0] * N

    for k in range(N // 2):
        w_odd = w * odd[k]
        y[k] = even[k] + w_odd
        y[k + N // 2] = even[k] - w_odd
        w *= w_n
    return y

def IFFT(x):
    N = len(x)
    if N <= 1: return x
    
    even = IFFT(x[0::2])
    odd = IFFT(x[1::2])

    w_n = exp(2j * pi / N)
    w = 1 + 0j
    y = [0] * N

    for k in range(N // 2):
        w_odd = w * odd[k]
        y[k] =          even[k] + w_odd
        y[k + N // 2] = even[k] - w_odd
        w *= w_n
    return [val / 2 for val in y]


def poly_mul(P, Q):
    # pad to length of next power of 2
    n = len(P) + len(Q) - 1
    n = 1 << (n - 1).bit_length()  
    P = P + [0] * (n - len(P))
    Q = Q + [0] * (n - len(Q))

    A = FFT(P)
    B = FFT(Q)

    C = [A[i] * B[i] for i in range(n)]
    c = IFFT(C)
    
    return remove_trailing_zeros([x.real for x in c])

def poly_subtract(P, Q):
    result = [0] * max(len(P), len(Q))
    for i in range(len(P)):
        result[i] += P[i]
    for i in range(len(Q)):
        result[i] -= Q[i]

    return remove_trailing_zeros(result)

def poly_mod_xk(f, k):
    return f[:k]

# -----------------------------------------------------------
#  Naive polynomial division 
# -----------------------------------------------------------

def poly_divide_naive(P, Q):
    n = len(P) - 1
    m = len(Q) - 1
    if n < m: return [0], P

    
    q = [0] * (n - m + 1)
    r = P[:]
    
    while len(r) >= len(Q):
        deg_diff = len(r) - len(Q)
        c = r[-1] / Q[-1] # leading coefficient for current division step
        q[deg_diff] = c
        
        # subtract polynomial c * Q(x) * x^deg_diff from r
        for i in range(len(Q)): 
            r[deg_diff + i] -= c * Q[i]

        r.pop() # at least one leading coeffecient is 0
        r = remove_trailing_zeros(r)
    q = remove_trailing_zeros(q)

    return q, r

# -----------------------------------------------------------
#   main alogrithm for polynomial division below here
# -----------------------------------------------------------

def poly_reciprocal_mod(P, n):
    g = [1 / P[0]]  
    r = (n - 1).bit_length()


    for i in range(r):
        m = 2**(i+1)
        t = poly_mod_xk(poly_mul(P, g), m)  
        t = poly_subtract([2], t) 
        g = poly_mod_xk(poly_mul(g, t), m)

    return poly_mod_xk(g, n)

def poly_divide_FFT(P, Q):
    n = len(P) - 1 # degree of P
    m = len(Q) - 1

    if n < m: return [0], P

    Q_rev = Q[::-1]
    P_rev = P[::-1]
    
    Q_rev_inv = poly_reciprocal_mod(Q_rev, n - m + 1)    
    q = poly_mod_xk(poly_mul(P_rev, Q_rev_inv), n - m + 1)   
    q = q[::-1]               
    r = poly_subtract(P, poly_mul(q, Q))                    

    return q, r

# -----------------------------------------------------------
#   example usage below here
# -----------------------------------------------------------

def benchmark():
    sizes = [100 * k for k in range(1, 100 + 1)]  # sizes of polynomials to test
    naive_times     = []
    fft_times       = []

    def time_algorithm(func, a, b):
        start = time.perf_counter()
        func(a[:], b[:])  
        return time.perf_counter() - start

    
    for n in sizes:
        print(f"dividing n={n}", end="")

        a = remove_trailing_zeros([random.randint(0, 10) for _ in range(n)]+[1])
        b = remove_trailing_zeros([random.randint(0, 10) for _ in range(n//2)]+[1])
        
        naive_times.append(time_algorithm(poly_divide_naive, a, b))
        fft_times.append(time_algorithm(poly_divide_FFT, a, b))  

        print(f": Done")

    plt.figure()
    plt.plot(sizes, naive_times)
    plt.plot(sizes, fft_times)

    plt.xlabel("Polynomial size (n) divisor size (n/2)")
    plt.ylabel("Time (seconds)")
    plt.title("Polynomial division comparison")
    plt.legend(["Naive O(n^2)", "FFT O(n log n)"])
    plt.show()

benchmark()