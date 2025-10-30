"""
fuzzy_weight_fucom_swara.py


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
        raise ValueError('Los componentes TFN deben ser > 0 para la regla de división.')
    return np.array([A[0] / B[2], A[1] / B[1], A[2] / B[0]])

# -----------------------------
# FUCOM (LP)
# -----------------------------

def f_fucom(criteria_order, chis):
    """Calcula FUCOM difuso.

    Inputs:
      - criteria_order: aceptado pero no usado
      - chis: lista de tuplas TFN de longitud n-1

    Output:
      - lista de TFNs: [(a,b,c), ...] (longitud n)

    Efectos secundarios: imprime TFNs, pesos defuzzificados (normalizados)
    y el valor de eta mínimo.

    Si la formulación con restricciones de transitividad (j -> j+2) resulta
    ser infactible, se intenta automáticamente resolver el modelo **sin**
    dichas restricciones y se avisa al usuario. Esto evita errores cuando las
    comparaciones generan inconsistencias numéricas.
    """
    n = len(chis) + 1

    def build_problem(include_transitivity=True):
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
            r1 = np.zeros(nv); r1[ix] = 1.0; r1[iy] = -k; r1[idx_eta] = -1.0
            r2 = np.zeros(nv); r2[ix] = -1.0; r2[iy] = k; r2[idx_eta] = -1.0
            A_ub.append(r1); b_ub.append(0.0)
            A_ub.append(r2); b_ub.append(0.0)

        # restricciones por comparación chi_j
        for j in range(n - 1):
            chi = chis[j]
            add_abs(idx(j, 'a'), chi[0], idx(j+1, 'c'))
            add_abs(idx(j, 'b'), chi[1], idx(j+1, 'b'))
            add_abs(idx(j, 'c'), chi[2], idx(j+1, 'a'))

        # restricciones por transitividad opcional (chi_j * chi_{j+1})
        if include_transitivity:
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

        A_ub_arr = np.array(A_ub)
        b_ub_arr = np.array(b_ub)

        # restricción de normalización: sum (1*a + 4*b + 1*c) = 6
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

        return cobj, A_ub_arr, b_ub_arr, A_eq, b_eq, bounds

    # intentar con transitividad; si es infactible, reintentar sin ella
    cobj, A_ub, b_ub, A_eq, b_eq, bounds = build_problem(include_transitivity=True)
    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        # si infeasible, reintentar sin transitividad y avisar
        if 'infeasible' in (res.message or '').lower():
            print('\nAviso: la formulación FUCOM con restricciones de transitividad resultó infactible. Reintentando sin dichas restricciones...')
            cobj, A_ub, b_ub, A_eq, b_eq, bounds = build_problem(include_transitivity=False)
            res2 = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res2.success:
                raise RuntimeError('LINPROG FUCOM falló incluso sin transitividad: ' + res2.message)
            else:
                res = res2
        else:
            raise RuntimeError('LINPROG FUCOM falló: ' + (res.message or 'sin mensaje'))

    x = res.x
    a = np.array([x[(3*i)+0] for i in range(n)])
    b = np.array([x[(3*i)+1] for i in range(n)])
    c = np.array([x[(3*i)+2] for i in range(n)])
    eta = float(max(x[3*n], 0.0))

    tfns = [ (float(a[i]), float(b[i]), float(c[i])) for i in range(n) ]

    # defuzzificados (normalizados)
    defuzz_raw = [ _defuzz_raw((a[i],b[i],c[i])) for i in range(n) ]
    s = sum(defuzz_raw)
    if s != 0.0:
        defuzz_norm = [float(v / s) for v in defuzz_raw]
    else:
        defuzz_norm = defuzz_raw[:]

    # imprimir resultados
    print('\nResultados F-FUCOM:')
    print(f'eta mínimo = {eta}')
    for i,(t,dn) in enumerate(zip(tfns, defuzz_norm), start=1):
        print(f"w_{i} = {_format_tfn(t)}, w_{i}def (norm) = {dn:.12f}")

    return tfns

# -----------------------------
# SWARA
# -----------------------------

def f_swara(criteria_order, chis):
    """Calcula SWARA difuso.

    Inputs:
      - criteria_order: aceptado pero no usado
      - chis: lista de TFNs de longitud n-1

    Output:
      - lista de TFNs SWARA

    Imprime TFNs y pesos defuzzificados (normalizados para lectura).
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

    print('\nResultados F-SWARA:')
    for i,(t,dn) in enumerate(zip(tfns, defuzz_norm), start=1):
        print(f"w_{i} = {_format_tfn(t)}, w_{i}def (norm) = {dn:.12f}")

    return tfns

# -----------------------------
# Pearson entre dos listas de TFNs
# -----------------------------

def pearson(fucom_tfns, swara_tfns):
    """Imprime y devuelve el coeficiente de correlación de Pearson entre
    los valores defuzzificados (usando (a+4b+c)/6) de dos listas de TFNs.

    Manejo especial de casos degenerados (desviación estándar cero):
      - Si ambas series son constantes y tienen el mismo valor, devolvemos r=1.0
        (perfecta concordancia en valores).
      - Si alguna de las series es constante pero los valores difieren, Pearson
        es indefinido; devolvemos `None` y mostramos un aviso.
      - Si ambas series tienen varianza > 0, usamos pearsonr de scipy como antes.
    """
    if len(fucom_tfns) != len(swara_tfns):
        raise ValueError('Las dos listas de TFNs deben tener la misma longitud')

    fucom_def = np.array([ _defuzz_raw(t) for t in fucom_tfns ], dtype=float)
    swara_def = np.array([ _defuzz_raw(t) for t in swara_tfns ], dtype=float)

    # tolerancia para considerar varianza nula
    tol = 1e-12
    std_f = float(np.std(fucom_def, ddof=0))
    std_s = float(np.std(swara_def, ddof=0))

    if std_f <= tol and std_s <= tol:
        # ambas constantes
        if abs(float(np.mean(fucom_def)) - float(np.mean(swara_def))) <= tol:
            print("\nAmbas series son constantes e iguales; definiendo r = 1.0")
            return 1.0
        else:
            print("\nAmbas series son constantes pero con distinto valor; Pearson indefinido.")
            return None
    elif std_f <= tol or std_s <= tol:
        # una es constante y la otra no -> indefinido
        print("\nUna de las series es constante (desviación estándar ~ 0); Pearson no está definido.")
        return None

    try:
        r_val, p_val = pearsonr(fucom_def, swara_def)
        print(f"\nCoeficiente de correlación de Pearson r = {r_val:.5f}")
        return float(r_val)
    except Exception as e:
        print('\nError al calcular Pearson:', e)
        return None