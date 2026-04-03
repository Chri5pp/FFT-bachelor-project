import random
from unittest import result

from numpy import half
random.seed(1234)

#------------------------------------------------------------
#           notes on multilinear polynomials
#------------------------------------------------------------

# multilinear polynomial of n variables has 2^n terms.

# theyre represented such that the ith index corresponds
# to the term where the jth variable is included if the
# jth bit of i is 1

# e.g. for n=3, the term at index 5 (0b101) corresponds to
# the term with x0 * x2, since the 0th and 2nd bits are set





#------------------------------------------------------------
#           MISC
#------------------------------------------------------------

def random_multilinear(n_args):
    return [random.randint(-9, 10) for _ in range(1 << n_args)]

def get_all_args(a_vals, b_vals):
    points = []
    for i in range(1 << len(a_vals)):
        point = []
        for j in range(len(a_vals)):
            point.append(b_vals[j] if (i >> j) & 1 else a_vals[j])
        points.append(point)
    return points





#------------------------------------------------------------
#           Evaluation
#------------------------------------------------------------

def evaluate_multilinear_naive(P, args):
    n_args = len(args)
    n_terms = len(P)
    
    result = 0
    for i in range(n_terms):
        term = P[i]
        for j in range(n_args):
            term *= args[j] if (i >> j) & 1 else 1
        result += term

    return result

def evaluate_multilinear_fast(P, args):
    n_args = len(args)
    P_start = 0
    P_end = len(P)

    def eval(P, args, P_start, P_end, n_args):
        if n_args == 0:
            return P[P_start]

        half = (P_end - P_start) // 2
        E1 = eval(P, args, P_start, P_start + half, n_args - 1)
        E2 = eval(P, args, P_start + half, P_end, n_args - 1)
        return E1 + E2 * args[n_args - 1]
    
    return eval(P, args, P_start, P_end, n_args)

# def evaluate_multilinear_fast(P, args):
#     n_args = len(args)
#     P_start = 0
#     P_end = len(P)
#     length = len(P)

#     # eval takes no arguments :)
#     def eval():
#         nonlocal n_args, P_start, P_end, length
#         if n_args == 0:
#             return P[P_start]

#         n_args -= 1
#         length >>= 1

#         P_end -= length
#         E1 = eval()
#         P_end += length

#         P_start += length
#         E2 = eval()
#         P_start -= length

#         length <<= 1
#         n_args += 1

#         return E1 + E2 * args[n_args - 1]
    
#     return eval()





#------------------------------------------------------------
#           Interpolation
#------------------------------------------------------------

def interpolate_multilinear_binary_fast(points): # assumes are evaluated at [1,0,0], [0,1,0], [1,1,0]... 
    n_terms = len(points)

    if n_terms == 1:
        return points
    
    # P = P0 when X_n = 0 and P = P1 when X_n = 1

    # P = X_n * P1 + (1 - X_n) * P0 <=>
    # P = P0 + X_n * (P1 - P0)

    # thus we get P0 from the first half of the points and P1 from the second half

    half_terms = n_terms // 2
    P0 = interpolate_multilinear_binary_fast(points[:half_terms]) # X_n = 0
    P1 = interpolate_multilinear_binary_fast(points[half_terms:]) # X_n = 1

    res = [0] * n_terms

    # the first half coeffecients are P1
    # and the second half coeffecients are P1 - P0 since theyre multiplied by X_n
    # (only last coeffecients are multiplied by the last variable)

    for i in range(half_terms):
        res[i] = P0[i]
        res[i + half_terms] = P1[i] - P0[i]

    return res

def interpolate_multilinear_fast(points, a_vals, b_vals):
    n_terms = len(points)

    if n_terms == 1:
        return points
    
    # P = P0 when X_n = a_n and P = P1 when X_n = b_n

    # P = (X_n - a_n) / (b_n - a_n) * P1 + (b_n - X_n) / (b_n - a_n) * P0 <=>
    # P = (b_nP_0 - a_nP1) / (b_n - a_n) + X_n * (P1 - P0) / (b_n - a_n)

    # thus we get P0 from the first half of the points and P1 from the second half

    half = n_terms // 2
    P0 = interpolate_multilinear_fast(points[:half], a_vals[:-1], b_vals[:-1]) # X_n = a_n
    P1 = interpolate_multilinear_fast(points[half:], a_vals[:-1], b_vals[:-1]) # X_n = b_n

    res = [0] * n_terms
    recipricol_divisor = 1 / (b_vals[-1] - a_vals[-1])
    for i in range(half):
        res[i] = (b_vals[-1] * P0[i] - a_vals[-1] * P1[i]) * recipricol_divisor
        res[i + half] = (P1[i] - P0[i]) * recipricol_divisor

    return res





#------------------------------------------------------------
#           multiplication
#------------------------------------------------------------

def multiply_multilinear_naive(P, Q):
    # asume P and Q only take 1 or 0 as inputs
    # such that X_i^2 = X_i

    n_args = len(P).bit_length() - 1
    n_terms = len(P)

    res = [0] * n_terms

    for i in range(n_terms):
        for j in range(n_terms):
            # the term at index k = i | j is the product of the terms at index i and j
            # Since X_i^2 = X_i the products just becomes union of variables
            k = i | j
            res[k] += P[i] * Q[j]

    return res

def multiply_multilinear_fast(P, Q):
    # asume P and Q only take 1 or 0 as inputs
    # such that X_i^2 = X_i

    n_args = len(P).bit_length() - 1
    n_terms = len(P)

    args = get_all_args([0] * n_args, [1] * n_args)
    
    P_points = [evaluate_multilinear_fast(P, arg) for arg in args]
    Q_points = [evaluate_multilinear_fast(Q, arg) for arg in args]
    
    points = [P_points[i] * Q_points[i] for i in range(n_terms)]

    # interpolate to over all points to recover the coefficients
    return interpolate_multilinear_binary_fast(points)





#------------------------------------------------------------
#           Tests
#------------------------------------------------------------
def test_evaluation():
    poly = random_multilinear(4)
    args = [1, 0, 1, 1]

    naive_result = evaluate_multilinear_naive(poly, args)
    fast_result = evaluate_multilinear_fast(poly, args)

    print("Coefficients:", poly)
    print("Naive evaluation result:", naive_result)
    print("Fast evaluation result:", fast_result)

def test_multilinear_interpolation():
    poly = random_multilinear(3)
    a_n = [0,3,0]
    b_n = [1,2,6]

    # evaluate at all points in the hypercube defined by a_n and b_n
    args = get_all_args(a_n, b_n)
    points = [evaluate_multilinear_fast(poly, arg) for arg in args]

    # interpolate to over all points to recover the coefficients
    recovered_poly = interpolate_multilinear_fast(points, a_n, b_n)

    print("Original coefficients:", poly)
    print("Recovered coefficients:", recovered_poly)

def test_multilinear_multiplication():
    poly1 = random_multilinear(3)
    poly2 = random_multilinear(3)

    product_naive = multiply_multilinear_naive(poly1, poly2)
    product_fast = multiply_multilinear_fast(poly1, poly2)

    print("Poly 1 coefficients:", poly1)
    print("Poly 2 coefficients:", poly2)
    print("Naive product coefficients:", product_naive)
    print("Fast product coefficients:", product_fast)

test_evaluation()
#test_multilinear_interpolation()
#test_multilinear_multiplication()