import random
random.seed(1234)

#------------------------------------------------------------
#           MISC
#------------------------------------------------------------

def random_multilinear(n_args):
    return [random.randint(-9, 10) for _ in range(1 << n_args)]

def pointwise_multiply(P, Q):
    return [P[i] * Q[i] for i in range(len(P))]

#------------------------------------------------------------
#           Evaluation
#------------------------------------------------------------

def evaluate_multilinear_fast(P):
    n_terms = len(P)

    if n_terms == 1:
        return P
    
    half = n_terms // 2
    Q0 = evaluate_multilinear_fast(P[:half])
    Q1 = evaluate_multilinear_fast(P[half:])

    result = [0] * n_terms

    for i in range(half):
        result[i]        = Q0[i]
        result[i + half] = Q0[i] + Q1[i]

    return result

#------------------------------------------------------------
#           Interpolation
#------------------------------------------------------------

def interpolate_multilinear_fast(points):
    n_terms = len(points)

    if n_terms == 1:
        return points
    
    half_terms = n_terms // 2
    P0 = interpolate_multilinear_fast(points[:half_terms]) 
    P1 = interpolate_multilinear_fast(points[half_terms:])

    res = [0] * n_terms


    for i in range(half_terms):
        res[i] = P0[i]
        res[i + half_terms] = P1[i] - P0[i]

    return res

#------------------------------------------------------------
#           multiplication
#------------------------------------------------------------

def multiply_multilinear_fast(P, Q):
    P_points = evaluate_multilinear_fast(P)
    Q_points = evaluate_multilinear_fast(Q)

    points = pointwise_multiply(P_points, Q_points)

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
            conv = multiply_multilinear_fast(P_deg[i], Q_deg[j])

            for k in range(N):
                res[k] += conv[k] if k.bit_count() == i + j else 0

    return res


P = [0, 1, 2, 3, 4, 5, 6, 7]
print("Results of different methods:")

print("O(4^n):    ", naive_subset_convolution(P, P))
print("O(3^n):    ", slow_subset_convolution(P, P))
print("O(n^3 2^n):", fast_subset_convolution(P, P))
