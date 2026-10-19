# Paper Example 16: GF13 missing monomial
# Run with SageMath. This file is self-contained.

EXAMPLE_NUMBER = 16
EXAMPLE_LABEL = 'Paper Example 16'
EXPECTED_SQUARES = (6, 6)
PROOF = 'schur'

from sage.all import GF, PolynomialRing, matrix, vector
from itertools import combinations, product

try:
    import numpy as np
except ImportError:
    np = None


def schur_dim(G):
    F = G.base_ring()
    rows = [[G[i, t] * G[j, t] for t in range(G.ncols())]
            for i in range(G.nrows()) for j in range(i, G.nrows())]
    return matrix(F, rows).rank()


def is_mds(G):
    k = G.nrows()
    return all(G.matrix_from_columns(list(S)).rank() == k
               for S in combinations(range(G.ncols()), k))


def normals(G):
    F, k, n = G.base_ring(), G.nrows(), G.ncols()
    found = set()
    for S in combinations(range(n), k - 1):
        B = matrix(F, [[G[i, j] for i in range(k)] for j in S])
        v = B.right_kernel().basis()[0]
        first = next(x for x in v if x != 0)
        found.add(tuple(v / first))
    return [vector(F, v) for v in found]


def search_beta(G, wanted):
    F, k = G.base_ring(), G.nrows()
    wanted = vector(F, wanted)
    first = next(x for x in wanted if x != 0)
    if first != 1:
        raise ValueError("The listed beta is not projective normal form.")
    M = normals(G)
    checked = 0

    # Integer batches make the search much faster over prime fields.
    if F.degree() == 1 and np is not None:
        q = int(F.order())
        normal_array = np.array([[int(x) for x in v] for v in M], dtype=np.int64)
        target = np.array([int(x) for x in wanted], dtype=np.int64)
        for pivot in range(k):
            total = q ** (k - pivot - 1)
            for start in range(0, total, 8192):
                stop = min(start + 8192, total)
                numbers = np.arange(start, stop, dtype=np.int64)
                cols = np.zeros((stop - start, k), dtype=np.int64)
                cols[:, pivot] = 1
                for pos in range(k - 1, pivot, -1):
                    cols[:, pos] = numbers % q
                    numbers //= q
                good = np.ones(stop - start, dtype=bool)
                for v in normal_array:
                    rows = np.flatnonzero(good)
                    if len(rows) == 0:
                        break
                    good[rows] = (cols[rows] @ v) % q != 0
                matches = np.flatnonzero(np.all(cols == target, axis=1))
                if len(matches):
                    t = int(matches[0])
                    if not good[t]:
                        raise ValueError("The listed beta is not admissible.")
                    return wanted, len(M), checked + t + 1
                checked += stop - start
    else:
        elements = list(F)
        for pivot in range(k):
            for tail in product(elements, repeat=k - pivot - 1):
                beta = vector(F, [0] * pivot + [1] + list(tail))
                checked += 1
                good = all(v.dot_product(beta) != 0 for v in M)
                if beta == wanted:
                    if not good:
                        raise ValueError("The listed beta is not admissible.")
                    return wanted, len(M), checked
    raise ValueError("The listed beta was not found in the projective search.")


def power_format(F, g, name):
    if F.degree() == 1:
        return lambda x: str(int(x))
    table = {F(0): "0"}
    for i in range(int(F.order()) - 1):
        table[g ** i] = "1" if i == 0 else name + "^" + str(i)
    if len(table) != F.order():
        raise ValueError("The displayed field element is not primitive.")
    return lambda x: table[F(x)]


def show_matrix(G, fmt):
    for row in G.rows():
        print("[" + "  ".join(fmt(x) for x in row) + "]")


def cauchy_rank(G):
    F, k = G.base_ring(), G.nrows()
    U = G.matrix_from_columns(list(range(k)))
    A = (U.inverse() * G).matrix_from_columns(list(range(k, G.ncols())))
    return matrix(F, [[1 / x for x in row] for row in A.rows()]).rank()


def run():
    F, generator, symbol, G, betas = build()
    fmt = power_format(F, generator, symbol)
    print(EXAMPLE_LABEL, "over GF(%s)" % F.order())
    print("G0:")
    show_matrix(G, fmt)
    assert is_mds(G), "G0 is not MDS."
    initial_square = schur_dim(G)
    print("Initial: [%s,%s,%s], Schur square = %s" %
          (G.ncols(), G.nrows(), G.ncols() - G.nrows() + 1, initial_square))
    if PROOF == "cauchy":
        print("Entrywise inverse rank =", cauchy_rank(G))
    if PROOF == "dual":
        print("Initial dual Schur square =", schur_dim(G.right_kernel().basis_matrix()))
    for step, wanted in enumerate(betas, 1):
        beta, normal_count, checked = search_beta(G, wanted)
        print("Step %s: %s normals, %s columns checked, beta = (%s)^T" %
              (step, normal_count, checked, ", ".join(fmt(x) for x in beta)))
        G = G.augment(matrix(F, [[x] for x in beta]))
    assert is_mds(G), "The final code is not MDS."
    final_square = schur_dim(G)
    if EXPECTED_SQUARES is not None:
        assert (initial_square, final_square) == EXPECTED_SQUARES, "Schur ranks do not match the source."
    print("G_final:")
    show_matrix(G, fmt)
    print("Final: [%s,%s,%s], Schur square = %s" %
          (G.ncols(), G.nrows(), G.ncols() - G.nrows() + 1, final_square))
    if PROOF == "dual":
        print("Final dual Schur square =", schur_dim(G.right_kernel().basis_matrix()))


def build():
    F = GF(13)
    alpha = [F(t) for t in range(6)]
    G = matrix(F, [[F(1) for t in alpha], alpha,
                   [t**3 for t in alpha]])
    return F, F(1), '', G, [(1,0,3),(1,1,0),(1,3,5),(1,4,5)]

if __name__ == '__main__':
    run()
