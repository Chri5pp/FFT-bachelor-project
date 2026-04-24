# ----------------------------------
#      Notes
# ----------------------------------

# these algorithms aren't assymptotically optimal since it uses
# naive multiplication and division of polynomials

# it is just to demonstrate the structure

# actually it should use the FFT optimised versions
# for optimal runtime O(n log^2 n)





# ----------------------------------
#      basic poly operations
# ----------------------------------

def eval_poly_horner(f, x):
    result = 0
    for coeff in reversed(f):
        result = result * x + coeff
    return result

def poly_add(P, Q):
    result = [0] * max(len(P), len(Q))
    for i in range(len(P)):
        result[i] += P[i]
    for i in range(len(Q)):
        result[i] += Q[i]

    while result and result[-1] == 0:
        result.pop()

    return result if result else [0]

def poly_mul(P, Q): # O(n^2)
    result = [0] * (len(P) + len(Q) - 1)
    for i in range(len(P)):
        for j in range(len(Q)):
            result[i + j] += P[i] * Q[j]
    return result

def poly_mod(P, Q): # O(n^2)
    P = P[:]

    while len(P) >= len(Q):
        # Q is monic (Q[-1] == 1) this is true for the polynomials in the tree since monic*monic = monic
        # therefor P[-1]/Q[-1] = P[-1]
        coeff = P[-1] 
        
        for i in range(len(Q)):
            P[len(P) - len(Q) + i] -= coeff * Q[i]
            
        while P and P[-1] == 0:
            P.pop()
    
    return P if P else [0]

def poly_derivative(P):
    return [i * P[i] for i in range(1, len(P))]





# ----------------------------------
#       Multipoint evaluation
# ----------------------------------

def build_tree(points):
    tree = []
    level = [[-u, 1] for u in points]
    tree.append(level)
    
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            next_level.append(poly_mul(level[i], level[i+1]))
        tree.append(next_level)
        level = next_level
    
    return tree

def evaluate_tree(f, tree, level, index):
    if level == 0:
        return [f[0]]   
    
    left_poly  = tree[level-1][2*index]
    right_poly = tree[level-1][2*index + 1]
    
    r0 = poly_mod(f, left_poly)
    r1 = poly_mod(f, right_poly)
    
    left_vals  = evaluate_tree(r0, tree, level-1, 2*index)
    right_vals = evaluate_tree(r1, tree, level-1, 2*index + 1)
    
    return left_vals + right_vals

def multipoint_evaluation(f, points):
    tree = build_tree(points)
    k = len(tree) - 1 
    return evaluate_tree(f, tree, k, 0)

print(multipoint_evaluation([1, 0, -1], [0, 1, 2, 3]))





# ----------------------------------
#       Interpolation
# ----------------------------------

def linear_combination(ci, tree, level, index):
    if level == 0:
        return [ci[0]]
    left_tree  = tree[level-1][2*index]
    right_tree = tree[level-1][2*index + 1]
    
    mid = len(ci) // 2
    r0 = linear_combination(ci[:mid], tree, level-1, 2*index)
    r1 = linear_combination(ci[mid:], tree, level-1, 2*index + 1)

    return poly_add(
        poly_mul(right_tree, r0),
        poly_mul(left_tree, r1)
    )

def interpolate(points, values):
    tree = build_tree(points)
    k = len(tree) - 1

    m = tree[-1][0]
    m_prime = poly_derivative(m)

    m_prime_vals = multipoint_evaluation(m_prime, points)
    si = [1 / v for v in m_prime_vals]

    ci = [values[i] * si[i] for i in range(len(values))]

    return linear_combination(ci, tree, k, 0)





# --------------------------
#       Test
# --------------------------

def title(msg, length = 40):
    print("\n" + "="*length)
    print("    " + msg)
    print("="*length)

def test_multipoint():
    title("Testing multipoint evaluation...")

    f = [1, 0, -1]  # 1 - x^2
    points = [0, 1, 2, 3]

    fast = multipoint_evaluation(f, points)
    slow = [eval_poly_horner(f, x) for x in points]

    print("fast:", fast)
    print("slow:", slow)



def test_interpolation():
    title("Testing interpolation...")

    points = [0, 1, 2, 3]
    f_true = [2, -1, 1]  # 2 - x + x^2

    values = multipoint_evaluation(f_true, points)
    f_rec = interpolate(points, values)

    print("original:", f_true)
    print("recovered:", f_rec)

test_multipoint()
test_interpolation()