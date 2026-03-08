import cmath

def recursive_FFT(x):
    N = len(x)
    if N <= 1: return x
    
    even = recursive_FFT(x[0::2])
    odd = recursive_FFT(x[1::2])

    w_n = cmath.exp(-2j * cmath.pi / N)
    w = 1 + 0j
    y = [0] * N

    for k in range(N // 2):
        w_odd = w * odd[k]
        y[k]          = even[k] + w_odd
        y[k + N // 2] = even[k] - w_odd
        w *= w_n
    return y

def recursive_IFFT(x):
    N = len(x)
    if N <= 1: return x
    
    even = recursive_IFFT(x[0::2])
    odd = recursive_IFFT(x[1::2])

    w_n = cmath.exp(2j * cmath.pi / N)
    w = 1 + 0j
    y = [0] * N

    for k in range(N // 2):
        w_odd = w * odd[k]
        y[k]          = even[k] + w_odd
        y[k + N // 2] = even[k] - w_odd
        w *= w_n
    return [val / 2 for val in y]

def bit_reversal_permutation(x):
    # this is done by counting and carrying bits in reverse order
    # this is better than computing the reverse of each index separately (O(n log n) vs O(n))
    N = len(x)
    i_reverse = 0
    for i in range(1, N):
        bit = N >> 1
        while i_reverse & bit != 0: # propergate carry to the right
            i_reverse ^= bit
            bit >>= 1
        i_reverse ^= bit

        if i < i_reverse: # avoid double swapping
            x[i], x[i_reverse] = x[i_reverse], x[i]
    return x

def iterative_FFT(x):
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

def iterative_IFFT(x):
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


A = [1, 2, 3, 4, 0, 0, 0, 0] + [1, 2, 3, 4, 0, 0, 0, 0] 
print("Iterative FFT(A):", [round(x.real) for x in iterative_FFT(A[:])])
print("recursive FFT(A):", [round(x.real) for x in recursive_FFT(A[:])])

