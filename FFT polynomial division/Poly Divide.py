from cmath import exp, pi
from math import ceil, log

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
        
        remove_trailing_zeros(r)
    remove_trailing_zeros(q)

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

def round_coefficients(poly):
    return [round(coef, 10) for coef in poly]

poly = [1, 2, 3, 4, 5]
divisor = [1, 2, 1]

q, r = poly_divide_FFT(poly, divisor)
q2, r2 = poly_divide_naive(poly, divisor)

print("division of", poly, "by", divisor)

print("\nFFT division:")
print("Quotient:", round_coefficients(q))
print("Remainder:", round_coefficients(r))

print("\nNaive division:")
print("Quotient:", round_coefficients(q2))
print("Remainder:", round_coefficients(r2))