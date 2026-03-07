import cmath

from numpy import real

def remove_trailing_zeros(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a

def FFT(x):
    N = len(x)
    if N <= 1: return x
    
    even = FFT(x[0::2])
    odd = FFT(x[1::2])

    w_n = cmath.exp(-2j * cmath.pi / N)
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

    w_n = cmath.exp(2j * cmath.pi / N)
    w = 1 + 0j
    y = [0] * N

    for k in range(N // 2):
        w_odd = w * odd[k]
        y[k] =          even[k] + w_odd
        y[k + N // 2] = even[k] - w_odd
        w *= w_n
    return [val / 2 for val in y]


def MultiplyPolynomialsFFT(a, b):
    # pad to length of next power of 2
    n = len(a) + len(b) - 1
    n = 1 << (n - 1).bit_length()  
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))

    A = FFT(a)
    B = FFT(b)

    C = [A[i] * B[i] for i in range(n)]
    c = IFFT(C)
    
    return remove_trailing_zeros([x.real for x in c])

def MultiplyPolynomialsNaive(a, b):
    c = [0] * (len(a) + len(b) - 1)

    for i in range(len(a)):
        for j in range(len(b)):
            c[i + j] += a[i] * b[j]

    return remove_trailing_zeros(c)


def multiply_polynomials_karatsuba(a, b):
    n = max(len(a), len(b))
    n = 1 << (n - 1).bit_length()  # pad to length of next power of 2

    a += [0] * (n - len(a))
    b += [0] * (n - len(b))

    c = karatsuba(a, b)

    return remove_trailing_zeros(c)

def karatsuba(a, b): 
    n = len(a)

    if n == 1:
        return [a[0] * b[0]]

    m = n // 2

    a_l = a[:m]
    a_r = a[m:]
    b_l = b[:m]
    b_r = b[m:]

    a_sum = [a_l[i] + a_r[i] for i in range(m)]
    b_sum = [b_l[i] + b_r[i] for i in range(m)]

    z_0 = karatsuba(a_l, b_l)
    z_1 = karatsuba(a_r, b_r)
    z_2 = karatsuba(a_sum, b_sum)

    y = [0] * (2 * n - 1)

    for i in range(n - 1):
        y[i]     += z_0[i]
        y[i + n] += z_1[i]
        y[i + m] += z_2[i] - z_0[i] - z_1[i]

    return y

a = [1, 2, 3, 4]
b = [5, 6]

 
print("Naive: O(n^2)")
print(MultiplyPolynomialsNaive(a[:], b[:])) 
print("Karatsuba: O(n^log2(3))")
print(multiply_polynomials_karatsuba(a[:], b[:]))
print("FFT: O(n log n)")
print(MultiplyPolynomialsFFT(a[:], b[:])) 

