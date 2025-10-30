"""
fuzzy_weight_fucom_swara.py

https://pdfs.semanticscholar.org/809b/d4cc55c66036264b463234d9e582471e9d31.pdf


Integrated F-FUCOM  + Fuzzy-SWARA (TFN) script.

Functions:
  - f_fucom(criteria_order, chis) -> Genetares fuzzy FUCOM weights[(a,b,c), ...]
    - Generates prints
  - f_swara(criteria_order, chis) -> Genetares fuzzy SWARA weights[(a,b,c), ...]
    - Generates prints
  - pearson(fucom_tfns, swara_tfns) -> Compute and print pearson correlation coefficient


"""

import numpy as np
from scipy.optimize import linprog
from scipy.stats import pearsonr

# -----------------------------
# Helpers TFN
# -----------------------------

def _format_tfn(t):
    t = np.asarray(t, dtype=float)
    return f"({t[0]:.12f}, {t[1]:.12f}, {t[2]:.12f})"

def _defuzz_raw(tfn):
    # norma clásica (a + 4b + c) / 6
    a, b, c = tfn
    return float((a + 4.0 * b + c) / 6.0)

def _div_tfn(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if np.any(A <= 0) or np.any(B <= 0):
        raise ValueError('Check TFN negative numbers')
    return np.array([A[0] / B[2], A[1] / B[1], A[2] / B[0]])

# helper: reorder TFNs according to criteria_order (expects 1-based indices)
def _reorder(tfns, criteria_order):
    """
    tfns: list of TFN tuples of length n (ordered by importance desc, index 0 = top)
    criteria_order: array-like of length n with a permutation of 1..n
    Returns: list of TFNs reordered so that output[pos] is the TFN corresponding to
             the criterion with external index pos+1.
    """
    import numpy as _np
    criteria_order = _np.asarray(criteria_order, dtype=int)
    n = len(tfns)
    if criteria_order.shape[0] != n:
        raise ValueError("criteria_order must have same length as number of TFNs")
    # validate permutation 1..n
    if set(criteria_order.tolist()) != set(range(1, n+1)):
        raise ValueError("criteria_order must be a permutation of 1..n (1-based indices)")
    out = [None] * n
    for i, pos in enumerate(criteria_order):
        out[pos - 1] = tfns[i]
    return out


# -----------------------------
# FUCOM (LP)
# -----------------------------

def f_fucom(criteria_order, chis):
    """Computes fuzzy FUCOM

    Inputs:
      - criteria_order: From the most relevant one to the least relevant one
      - chis: list of comparison wrt the precious criteria

    Output:
      - TFNs: [(a,b,c), ...] 

    """
    n = len(chis) + 1

    # variables: a1,b1,c1, a2,b2,c2, ..., eta => 3*n + 1
    nv = 3 * n + 1
    def idx(i, var):
        base = 3 * i
        return base + {'a':0, 'b':1, 'c':2}[var]
    idx_eta = 3 * n

    cobj = np.zeros(nv)
    cobj[idx_eta] = 1.0

    A_ub = []
    b_ub = []

    def add_abs(ix, k, iy):

        r1 = np.zeros(nv)
        r1[ix] = 1.0
        r1[iy] = -k
        r1[idx_eta] = -1.0

        r2 = np.zeros(nv)
        r2[ix] = -1.0
        r2[iy] = k
        r2[idx_eta] = -1.0

        A_ub.append(r1); b_ub.append(0.0)
        A_ub.append(r2); b_ub.append(0.0)

    # Restrictionsof comparison chi_j
    for j in range(n - 1):
        chi = chis[j]
        add_abs(idx(j, 'a'), chi[0], idx(j+1, 'c'))
        add_abs(idx(j, 'b'), chi[1], idx(j+1, 'b'))
        add_abs(idx(j, 'c'), chi[2], idx(j+1, 'a'))

    # Restrictions of transitivity (chi_j * chi_{j+1})
    for j in range(n - 2):
        chi = chis[j]
        chi1 = chis[j+1]
        add_abs(idx(j, 'a'), chi[0]*chi1[0], idx(j+2, 'c'))
        add_abs(idx(j, 'b'), chi[1]*chi1[1], idx(j+2, 'b'))
        add_abs(idx(j, 'c'), chi[2]*chi1[2], idx(j+2, 'a'))

    # orden a <= b <= c
    for i in range(n):
        r = np.zeros(nv); r[idx(i,'a')] = 1.0; r[idx(i,'b')] = -1.0
        A_ub.append(r); b_ub.append(0.0)
        r2 = np.zeros(nv); r2[idx(i,'b')] = 1.0; r2[idx(i,'c')] = -1.0
        A_ub.append(r2); b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Normalization restrictions sum (1*a + 4*b + 1*c) = 6
    A_eq = np.zeros((1, nv))
    for i in range(n):
        A_eq[0, idx(i,'a')] = 1.0
        A_eq[0, idx(i,'b')] = 4.0
        A_eq[0, idx(i,'c')] = 1.0
    b_eq = np.array([6.0])

    # bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, None))   # a_i >= 0
        bounds.append((None, None))  # b_i libre
        bounds.append((None, None))  # c_i libre
    bounds.append((0.0, None))      # eta >= 0

    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError('LINPROG FUCOM falló: ' + res.message)

    x = res.x
    a = np.array([x[idx(i,'a')] for i in range(n)])
    b = np.array([x[idx(i,'b')] for i in range(n)])
    c = np.array([x[idx(i,'c')] for i in range(n)])
    eta = float(max(x[idx_eta], 0.0))

    tfns = [ (float(a[i]), float(b[i]), float(c[i])) for i in range(n) ]

    # Defuzzification
    defuzz_raw = [ _defuzz_raw((a[i],b[i],c[i])) for i in range(n) ]
    s = sum(defuzz_raw)
    if s != 0.0:
        defuzz_norm = [float(v / s) for v in defuzz_raw]
    else:
        defuzz_norm = defuzz_raw[:]

    tfns = [ (float(a[i])/s, float(b[i])/s, float(c[i])/s) for i in range(n) ]
   
    # Reorder weights wrt criteria_order input
    tfns_out = _reorder(tfns, criteria_order)
    defuzz_raw_out = _reorder(defuzz_raw, criteria_order)
    defuzz_norm_out = _reorder(defuzz_norm, criteria_order)

    # Print
    print('\nResults F-FUCOM:')
    print(f'eta min = {eta}')
    for i,(t,dn) in enumerate(zip(tfns_out, defuzz_norm_out), start=1):
        print(f"w_{i} = {_format_tfn(t)}, w_{i}def (norm) = {dn:.12f}")

    return tfns_out

# -----------------------------
# SWARA
# -----------------------------

def f_swara(criteria_order, chis):
    """Computes fuzzy Swara weights

    Inputs:
      - criteria_order: Order of criteria from the most relevant one to the least relevant one
      - chis: list of comparisons

    Output:
      - TFN fuzzy swara weights

    
    """
    n = len(chis) + 1
    q_list = [np.array([1.0, 1.0, 1.0], dtype=float)]
    for chi in chis:
        q_prev = q_list[-1]
        q_next = _div_tfn(q_prev, chi)
        q_list.append(q_next)

    Q = np.sum(q_list, axis=0)
    w_list = [ _div_tfn(q, Q) for q in q_list ]

    tfns = [ (float(w[0]), float(w[1]), float(w[2])) for w in w_list ]

    defuzz_raw = [ _defuzz_raw(w) for w in w_list ]
    s = sum(defuzz_raw)
    if s != 0.0:
        defuzz_norm = [float(v / s) for v in defuzz_raw]
    else:
        defuzz_norm = defuzz_raw[:]

    tfns = [ (float(w[0])/s, float(w[1])/s, float(w[2])/s) for w in w_list ]

   # Reorder weights wrt criteria_order input
    tfns_out = _reorder(tfns, criteria_order)
    defuzz_raw_out = _reorder(defuzz_raw, criteria_order)
    defuzz_norm_out = _reorder(defuzz_norm, criteria_order)

    print('\nResults F-SWARA:')
    for i,(t,dn) in enumerate(zip(tfns_out, defuzz_norm_out), start=1):
        print(f"w_{i} = {_format_tfn(t)}, w_{i}def (norm) = {dn:.12f}")

    return tfns_out

# -----------------------------
# Pearson entre dos listas de TFNs
# -----------------------------

def pearson(fucom_tfns, swara_tfns):
    """Computes and prints pearson correlation coeficient of the defuzzified weigths
    """
    if len(fucom_tfns) != len(swara_tfns):
        raise ValueError('Both list must have the same number of criterion')

    fucom_def = [ _defuzz_raw(t) for t in fucom_tfns ]
    swara_def = [ _defuzz_raw(t) for t in swara_tfns ]

    try:
        r_val, p_val = pearsonr(np.array(fucom_def), np.array(swara_def))
        print(f"Pearson correlation coefficient r = {r_val:.5f}")
        return float(r_val)
    except Exception as e:
        print('Error when computing pearson cor.coefficient:', e)
        return None
