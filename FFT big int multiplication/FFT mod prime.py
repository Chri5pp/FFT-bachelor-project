def remove_trailing_zeros(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a

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


def multiplyPolynomials_mod_prime(a, b, root, p):
    # pad to length of next power of 2 (that fits the result)
    n = len(a) + len(b) - 1
    n = 1 << (n - 1).bit_length()  
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))

    w = get_nth_root_of_unity(n, p, root) # TODO: check if length of n is valid for using this root (n is not too large >2^23)
    A = mod_FFT(a, w, p)
    B = mod_FFT(b, w, p)

    C = [(A[i] * B[i]) % p for i in range(n)]
    c = mod_ifft(C, w, p)

    return remove_trailing_zeros(c)


def multiply_big_positive_integers(x, y):
    # this choice works up to 2^23 nth roots of unity for radix-2 FFT
    # found numbers at: https://codeforces.com/blog/entry/75326 (last visited 4/3/2026)
    p = 998244353       # 119 * 2^23 + 1
    root = 3            # 3^{998244352} = 1 (mod 998244353)

    print(f"Multiplying x and y:\n  {x}\n  {y}")

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
    print(f"\nRepresenting x and y in base: {base}:")
    print(f"x: {a}")
    print(f"y: {b}")


    # check that predefined prime is large enough to not cause unwanted overflow
    max_coeefficient = base - 1
    min_prime = (max_coeefficient ** 2) * min(len(a), len(b))
    if p <= min_prime:
        raise Exception(f"Prime {p} is too small. Minimum required prime is: {min_prime}")

    c = multiplyPolynomials_mod_prime(a, b, root, p) # where the magic happens
    print(f"\nmultiplying polynomials gives:\n  {c}")

    # propegate carries
    carry = 0
    for i in range(len(c)):
        c[i] += carry
        carry = c[i] >> base_shift
        c[i] &= base - 1
    if carry:
        c.append(carry)

    print(f"\npropegating carries gives:\n  {c}")
    # reconstruct c to big python number
    z = 0
    for i in range(len(c)):
        z += c[i] << (base_shift * i) 

    print(f"\nreconstring the final value gives:\n  {z}")
    print(f"expected result:\n  {x * y}")
    return(z)
    

multiply_big_positive_integers(12345678912345789123456789123457891234567891234578912345678912345789, 987654321987654321123456789123457891234567891234578912345678912345789)
