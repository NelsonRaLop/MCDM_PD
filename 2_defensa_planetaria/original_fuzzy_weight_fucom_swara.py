"""
combined_fucom_swara.py

Integrated F-FUCOM  + Fuzzy-SWARA (TFN) script.

"""

import numpy as np
from scipy.optimize import linprog
from scipy.stats import pearsonr

# --------------------------
# TFN helper utilities
# --------------------------

def add_tfn(A, B):
    """Component-wise addition of TFNs A and B (arrays or tuples of length 3)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return A + B

def div_tfn(A, B):
    """
    Division of TFN A by TFN B with the convention:
      (a1,a2,a3) ÷ (b1,b2,b3) = (a1/b3, a2/b2, a3/b1)
    Requires strictly positive components for stability.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if np.any(B <= 0) or np.any(A <= 0):
        raise ValueError("All TFN components must be > 0 for this division convention.")
    return np.array([A[0] / B[2], A[1] / B[1], A[2] / B[0]])

def weighted_defuzz(A):
    """Defuzzify TFN A using (a + 4*b + c) / 6."""
    A = np.asarray(A, dtype=float)
    return float((A[0] + 4.0 * A[1] + A[2]) / 6.0)

def format_tfn(t):
    """Pretty formatting for TFN tuples."""
    t = np.asarray(t, dtype=float)
    return f"({t[0]:.8f}, {t[1]:.8f}, {t[2]:.8f})"

# --------------------------
# FUCOM implementation
# Implemented as https://pdfs.semanticscholar.org/809b/d4cc55c66036264b463234d9e582471e9d31.pdf
# --------------------------

def solve_min_eta(n, chis):
    """
    Solve F-FUCOM LP for n criteria given chis of length n-1.
    Returns (a, b, c, eta, res) where a,b,c are arrays length n.
    """
    if len(chis) != n - 1:
        raise ValueError("chis must have length n-1")

    # variable indexing: for i in 0..n-1 -> (a_i, b_i, c_i) at positions (3*i, 3*i+1, 3*i+2)
    nv = 3 * n + 1  # +1 for eta
    def idx(i, var):
        base = 3 * i
        return base + {"a":0, "b":1, "c":2}[var]

    idx_eta = 3 * n

    # Objective: minimize eta
    cobj = np.zeros(nv)
    cobj[idx_eta] = 1.0

    A_ub = []
    b_ub = []

    def add_abs_constraint(ix, k, iy):
        # ix - k*iy - eta <= 0
        # -ix + k*iy - eta <= 0
        row1 = np.zeros(nv); row1[ix] = 1.0; row1[iy] = -k; row1[idx_eta] = -1.0
        row2 = np.zeros(nv); row2[ix] = -1.0; row2[iy] = k; row2[idx_eta] = -1.0
        A_ub.append(row1); b_ub.append(0.0)
        A_ub.append(row2); b_ub.append(0.0)

    # primary neighbor constraints (j vs j+1) using chis[j] for j=0..n-2
    for j in range(n - 1):
        chi_j = chis[j]
        add_abs_constraint(idx(j, "a"), chi_j[0], idx(j+1, "c"))
        add_abs_constraint(idx(j, "b"), chi_j[1], idx(j+1, "b"))
        add_abs_constraint(idx(j, "c"), chi_j[2], idx(j+1, "a"))

    # next-next constraints (j vs j+2)
    for j in range(n - 2):
        chi_j = chis[j]
        chi_jp1 = chis[j + 1]
        add_abs_constraint(idx(j, "a"), chi_j[0] * chi_jp1[0], idx(j+2, "c"))
        add_abs_constraint(idx(j, "b"), chi_j[1] * chi_jp1[1], idx(j+2, "b"))
        add_abs_constraint(idx(j, "c"), chi_j[2] * chi_jp1[2], idx(j+2, "a"))

    # order constraints a_i <= b_i <= c_i
    for i in range(n):
        row = np.zeros(nv); row[idx(i, "a")] = 1.0; row[idx(i, "b")] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        row2 = np.zeros(nv); row2[idx(i, "b")] = 1.0; row2[idx(i, "c")] = -1.0
        A_ub.append(row2); b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # equality: sum_i (a_i + 4*b_i + c_i) = 6  (so defuzzified weights sum to 1)
    A_eq = np.zeros((1, nv))
    for i in range(n):
        A_eq[0, idx(i, "a")] = 1.0
        A_eq[0, idx(i, "b")] = 4.0
        A_eq[0, idx(i, "c")] = 1.0
    b_eq = np.array([6.0])

    # bounds (a_i >=0, b_i and c_i free, eta >=0)
    bounds = []
    for i in range(n):
        bounds.append((0.0, None))   # a_i >= 0
        bounds.append((None, None))  # b_i free
        bounds.append((None, None))  # c_i free
    bounds.append((0.0, None))      # eta >= 0

    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return None, None, None, None, res

    x = res.x
    a = np.array([x[idx(i, "a")] for i in range(n)])
    b = np.array([x[idx(i, "b")] for i in range(n)])
    c = np.array([x[idx(i, "c")] for i in range(n)])
    eta_val = float(max(x[idx_eta], 0.0))

    return a, b, c, eta_val, res

def f_fucom(chis, n):
    """
    Wrapper for FUCOM. Inputs:
      - chis: list length n-1 of TFN triples
      - n: number of criteria
    Returns:
      - w_tfns: list of TFN tuples (a,b,c)
      - w_crisp: list of defuzzified weights normalized to sum 1
      - eta_val: minimal eta value
    """
    a, b, c, eta_val, res = solve_min_eta(n, chis)
    if a is None:
        raise RuntimeError("FUCOM LP failed: " + str(res.message if res is not None else "no result"))
    w_tfns = [ (float(a[i]), float(b[i]), float(c[i])) for i in range(n) ]
    w_crisp = [ weighted_defuzz((a[i], b[i], c[i])) for i in range(n) ]
    total = sum(w_crisp)
    if total != 0.0:
        w_crisp = [float(v / total) for v in w_crisp]
    return w_tfns, w_crisp, float(eta_val)

# --------------------------
# Fuzzy SWARA implementation
# Implemented as https://link.springer.com/article/10.1007/s00170-016-9880-x
# --------------------------

def f_swara(chis, n):
    """
    Fuzzy SWARA implementation.
    Inputs:
      - chis: list length n-1 of TFN triples
      - n: number of criteria
    Returns:
      - w_tfns: list of TFN tuples (a,b,c) (normalized by fuzzy Q)
      - w_crisp: list of defuzzified weights normalized to sum 1
    """
    q_list = [np.array([1.0, 1.0, 1.0], dtype=float)]
    for chi in chis:
        q_prev = q_list[-1]
        q_next = div_tfn(q_prev, chi)
        q_list.append(q_next)
    Q = np.sum(q_list, axis=0)
    w_list = [div_tfn(q, Q) for q in q_list]
    w_tfns = [ (float(w[0]), float(w[1]), float(w[2])) for w in w_list ]
    w_crisp = [ weighted_defuzz(w) for w in w_list ]
    total = sum(w_crisp)
    if total != 0.0:
        w_crisp = [float(v / total) for v in w_crisp]
    return w_tfns, w_crisp







# --------------------------
# MAIN!!!!!!!!!!!!!!!!!!!!!!
# --------------------------







if __name__ == "__main__":
    # Example input (modify as needed). chis length = n - 1
    chis = [
        #(1.400, 1.500, 1.667),
        #(1.222, 1.250, 1.286),
        #(1.667, 2.000, 2.500),
    # Comparison wrt the previous one
(0.67,1,1.5),
(1,2,3.73),
(1,1.5,2.33),
(1,1.33,1.8),
(0.78,1,1.29) 

    ]

    # number of criteria
    n = len(chis) + 1

    # compute FUCOM weights
    w_tfns_fucom, w_crisp_fucom, eta_val = f_fucom(chis, n)

    # compute Fuzzy SWARA weights
    w_tfns_swara, w_crisp_swara = f_swara(chis, n)

    # Pearson correlation between defuzzified crisp weight vectors
    try:
        r_val, p_val = pearsonr(np.array(w_crisp_fucom), np.array(w_crisp_swara))
    except Exception:
        corr = np.corrcoef(np.array(w_crisp_fucom), np.array(w_crisp_swara))
        r_val = float(corr[0,1]) if corr.shape == (2,2) else float("nan")
        p_val = float("nan")

    # Print results (Spanish style labels as in your example)
    print("\nResultados F-FUCOM:")
    print(f"eta mínimo = {eta_val}")
    for i, (tf, crisp) in enumerate(zip(w_tfns_fucom, w_crisp_fucom), start=1):
        print(f"w_{i} = {format_tfn(tf)}, w_{i}def = {crisp:.12f}")

    print("\nResultados F-SWARA:")
    for i, (tf, crisp) in enumerate(zip(w_tfns_swara, w_crisp_swara), start=1):
        print(f"w_{i} = {format_tfn(tf)}, w_{i}def = {crisp:.12f}")

    print(f"\nCoeficiente de correlación de Pearson r = {r_val:.5f}")
