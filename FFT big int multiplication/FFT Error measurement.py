import cmath
from random import random
import time
import random
from matplotlib import pyplot as plt

random.seed(1234)

def remove_trailing_zeros(a, eps=1e-9):
    while len(a) > 1 and abs(a[-1]) < eps:
        a.pop()
    return a

# def remove_trailing_zeros(a):
#     while len(a) > 1 and a[-1] == 0:
#         a.pop()
#     return a

def mod_prime_inverse(a, p):
    return pow(a, p-2, p)

def get_nth_root_of_unity(n, p, root):
    w = pow(root, (p - 1) // n, p)
    return w

def bit_reversal_permutation(x):
    N = len(x)
    i_reverse = 0
    for i in range(1, N):
        bit = N >> 1
        while i_reverse & bit:
            i_reverse ^= bit
            bit >>= 1
        i_reverse ^= bit

        if i < i_reverse: # avoid double swapping
            x[i], x[i_reverse] = x[i_reverse], x[i]
    return x

def mod_FFT(x, w, p):
    N = len(x)
    if N  == 1: return x
    x = bit_reversal_permutation(x)

    # precompute twiddle factors
    W = [0] * (N // 2)

    W[0] = 1
    for k in range(1, N // 2):
        W[k] = W[k-1] * w % p

    length = 2
    step_size = N >> 1
    while length <= N:
        half_length = length >> 1
        for i in range(0, N, length):
            step = 0
            for k in range(half_length):
                t1 = x[i + k]
                t2 = W[step] * x[i + k + half_length] % p
                x[i + k]               = (t1 + t2) % p
                x[i + k + half_length] = (t1 - t2) % p
                step += step_size
        length <<= 1
        step_size >>= 1

    return x

def mod_ifft(a, w, p):
    # the inverse FFT is just the FFT with w replaced by w^{-1} and then scaled by n^{-1}
    n = len(a)
    w_inv = mod_prime_inverse(w, p)
    y = mod_FFT(a, w_inv, p)
    n_inv = mod_prime_inverse(n, p)
    return [(x * n_inv) % p for x in y]

def imaginary_FFT(x):
    N = len(x)
    x = bit_reversal_permutation(x) 
    W = [cmath.exp(-2j * cmath.pi * k / N) for k in range(N//2)] # precompute twiddle factors

    length = 2
    step = N // length
    while length <= N:
        
        for i in range(0, N, length):
            for k in range(length // 2):
                even  = x[i + k]
                w_odd = W[k * step] * x[i + k + length // 2] 
                x[i + k]               = even + w_odd
                x[i + k + length // 2] = even - w_odd
        length *= 2
        step //= 2

    return x

def imaginary_IFFT(x):
    N = len(x)
    x = bit_reversal_permutation(x) 
    W = [cmath.exp(2j * cmath.pi * k / N) for k in range(N//2)] # precompute twiddle factors

    length = 2
    step = N // length
    while length <= N:
        step = N // length
        for i in range(0, N, length):
            for k in range(length // 2):
                even  = x[i + k]
                w_odd = W[k * step] * x[i + k + length // 2] 
                x[i + k]               = even + w_odd
                x[i + k + length // 2] = even - w_odd
        length *= 2
        step //= 2

    return [val / N for val in x]

def multiply_polynomials_mod_prime(a, b, root, p):
    # pad to length of next power of 2 (that fits the result)
    n = len(a) + len(b) - 1
    n = 1 << (n - 1).bit_length()  
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))

    if n > 1 << 23:
        raise Exception(f"length of resulting polynomial: {n} -is too large for using the predefined root. Maximum allowed length is: {1 << 23}")

    w = get_nth_root_of_unity(n, p, root) 
    A = mod_FFT(a, w, p)
    B = mod_FFT(b, w, p)

    C = [(A[i] * B[i]) % p for i in range(n)]
    c = mod_ifft(C, w, p)

    return remove_trailing_zeros(c)

def multiply_polynomials(a, b):
    # pad to length of next power of 2 (that fits the result)
    n = len(a) + len(b) - 1
    n = 1 << (n - 1).bit_length()  
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))

    A = imaginary_FFT(a)
    B = imaginary_FFT(b)

    C = [(A[i] * B[i]) for i in range(n)]
    c = imaginary_IFFT(C)

    return remove_trailing_zeros([x.real for x in c])

def multiply_big_positive_integers(x, y):
    # this choice works up to 2^23 nth roots of unity for radix-2 FFT
    # found numbers at: https://codeforces.com/blog/entry/75326 (last visited 4/3/2026)
    p = 998244353       # 119 * 2^23 + 1
    root = 3            # 3^{998244352} = 1 (mod 998244353)

    # represent big numbers as polynomials
    base_shift = max(x, y).bit_length().bit_length()
    base = 1 << base_shift  # about log2 of largest number (and a power of 2 to allow more bitwise operations)

    def split_number(n):
        coef = [] 
        while n > 0:
            coef.append(n & (base - 1))
            n >>= base_shift
        return coef

    a = split_number(x)
    b = split_number(y)


    # check that predefined prime is large enough to not cause unwanted overflow
    max_coeefficient = base - 1
    min_prime = (max_coeefficient ** 2) * min(len(a), len(b))
    if p <= min_prime:
        raise Exception(f"Prime {p} is too small. Minimum required prime is: {min_prime}")

    c = multiply_polynomials_mod_prime(a, b, root, p) # where the magic happens
    # propegate carries
    carry = 0
    for i in range(len(c)):
        c[i] += carry
        carry = c[i] >> base_shift
        c[i] &= base - 1
    if carry:
        c.append(carry)

    z = 0
    for i in range(len(c)):
        z += c[i] << (base_shift * i) 


    return z
    
def multiply_big_positive_integers_2(x, y):
    # represent big numbers as polynomials
    base_shift = max(x, y).bit_length().bit_length()
    base = 1 << base_shift  # about log2 of largest number (and a power of 2 to allow more bitwise operations)

    def split_number(n):
        coef = [] 
        while n > 0:
            coef.append(n & (base - 1))
            n >>= base_shift
        return coef

    a = split_number(x)
    b = split_number(y)

    c = multiply_polynomials(a, b) # where the magic happens

    carry = 0
    for i in range(len(c)):
        c[i] += carry
        carry = c[i] // base
        c[i] %= base

    if carry:
        c.append(carry)

    z = 0
    for i in range(len(c)):
        z += c[i] * (base ** i)


    return z

def benchmark():
    sizes = [10 ** k for k in range(1, 100 + 1)] # max size of integers to multiply
    Error     = []

    for n in sizes:
        print(f"Finding error for n being 2^{n.bit_length()}", end="")

        err = 0
        samples = 10
        for _ in range(samples): 
            a = random.randint(1, n)
            b = random.randint(1, n)
            
            res1 = multiply_big_positive_integers(a, b)
            res2 = multiply_big_positive_integers_2(a, b)

            err += abs(res1 - res2) / abs(res1)

        Error.append(err / samples) # average relative error
        print(f": Done")

    plt.figure()
    plt.plot(sizes, Error)


    plt.xlabel("Size of random number (0 to n)")
    plt.ylabel("Reletive error")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Integer multiplication error accumulation")
    plt.show()
    
benchmark()
