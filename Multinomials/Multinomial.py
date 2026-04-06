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





#------------------------------------------------------------
#           Evaluation
#------------------------------------------------------------

def evaluate_multilinear_naive(P, a_vals, b_vals):
    n_args = len(a_vals)
    n_terms = len(P)
    
    result = [0] * n_terms

    for k in range(n_terms): # individually calculate the result for all k points
        for i in range(n_terms):
            term = P[i]
            for j in range(n_args):
                if (i >> j) & 1:
                    arg = b_vals[j] if (k >> j) & 1 else a_vals[j] # arguments depends in the kth point we evaluate at
                    term *= arg
            result[k] += term

    return result

def evaluate_multilinear_fast(P, a_vals, b_vals):
    n_args = len(a_vals)
    n_terms = len(P)

    if n_args == 0:
        return P
    
    half = n_terms // 2
    Q0 = evaluate_multilinear_fast(P[:half], a_vals[:-1], b_vals[:-1])
    Q1 = evaluate_multilinear_fast(P[half:], a_vals[:-1], b_vals[:-1]) # multiplied by X_n

    result = [0] * n_terms

    # P = Q0 + X_n * Q1 
    # X_n = a_n for the first half of the points 
    # X_n = b_n for the second half of the points
    for i in range(half):
        result[i]        = Q0[i] + Q1[i] * a_vals[-1]
        result[i + half] = Q0[i] + Q1[i] * b_vals[-1]

    return result





#------------------------------------------------------------
#           Interpolation
#------------------------------------------------------------

def interpolate_multilinear_binary_fast(points):
     # assumes a_vals = [0,0...] and b_vals = [1,1,...]
     # no need for floating point division since the divisor is always 1
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
    for i in range(half):
        res[i]        = (b_vals[-1] * P0[i] - a_vals[-1] * P1[i]) / (b_vals[-1] - a_vals[-1])
        res[i + half] = (P1[i] - P0[i])                           / (b_vals[-1] - a_vals[-1])

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

    a_vals = [0] * n_args
    b_vals = [1] * n_args
    
    P_points = evaluate_multilinear_fast(P, a_vals, b_vals)
    Q_points = evaluate_multilinear_fast(Q, a_vals, b_vals)
    
    points = [P_points[i] * Q_points[i] for i in range(n_terms)]

    # interpolate to over all points to recover the coefficients
    return interpolate_multilinear_binary_fast(points)





#------------------------------------------------------------
#           Tests
#------------------------------------------------------------
def title(text, width=30):
    print("\n" + "=" * width)
    print(f"    {text}")
    print("=" * width)

def test_evaluation():
    title("Evaluation test")
    poly = random_multilinear(4)
    a_vals = [0, 0, 0, 0]
    b_vals = [1, 1, 1, 1]

    naive_result = evaluate_multilinear_naive(poly, a_vals, b_vals)
    fast_result = evaluate_multilinear_fast(poly, a_vals, b_vals)

    print("Coefficients:", poly)
    print("Naive evaluation result:", naive_result)
    print("Fast evaluation result: ", fast_result)

def test_multilinear_interpolation():
    title("Interpolation test")
    poly = random_multilinear(4)
    a_vals = [0, 0, 0, 0]
    b_vals = [1, 1, 1, 1]

    points = evaluate_multilinear_fast(poly, a_vals, b_vals)
    recovered_poly = interpolate_multilinear_fast(points, a_vals, b_vals)
    recovered_poly = [round(coef) for coef in recovered_poly] # round to nearest integer since we know the original coefficients are integers

    print("Original coefficients: ", poly)
    print("Recovered coefficients:", recovered_poly)

def test_multilinear_multiplication():
    title("Multiplication test")
    poly1 = random_multilinear(4)
    poly2 = random_multilinear(4)

    product_naive = multiply_multilinear_naive(poly1, poly2)
    product_fast = multiply_multilinear_fast(poly1, poly2)

    print("Poly 1 coefficients:", poly1)
    print("Poly 2 coefficients:", poly2)
    print("Naive product coefficients:", product_naive)
    print("Fast product coefficients: ", product_fast)

test_evaluation()
test_multilinear_interpolation()
test_multilinear_multiplication()