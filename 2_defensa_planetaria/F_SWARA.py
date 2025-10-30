import numpy as np

def div_tfn(A, B):
    """
    División de TFN con convención:
    (a1/b3, a2/b2, a3/b1)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    return np.array([A[0]/B[2], A[1]/B[1], A[2]/B[0]])

def add_tfn(A, B):
    """Suma de TFN componente a componente"""
    return np.array(A) + np.array(B)

def weighted_defuzz(A):
    """Defuzzificación con la norma (a+4b+c)/6"""
    A = np.array(A, dtype=float)
    return (A[0] + 4*A[1] + A[2]) / 6.0

def compute_fuzzy_swara(n, chis):
    if len(chis) != n-1:
        raise ValueError(f"Se esperaban {n-1} chi, pero se dieron {len(chis)}.")

    # q1
    q_list = [np.array([1.0, 1.0, 1.0])]
    # Calcular q2..qn
    for chi in chis:
        q_prev = q_list[-1]
        q_next = div_tfn(q_prev, chi)
        q_list.append(q_next)

    # Q = suma de q_j
    Q = np.sum(q_list, axis=0)

    # Pesos difusos normalizados
    w_list = [div_tfn(q, Q) for q in q_list]

    # Defuzzificación
    defuzz_list = [weighted_defuzz(w) for w in w_list]

    # Normalización defuzzificados
    defuzz_sum = np.sum(defuzz_list)
    norm_defuzz = np.array(defuzz_list) / defuzz_sum

    return q_list, Q, w_list, defuzz_list, norm_defuzz

if __name__ == "__main__":
    # Ejemplo
    n = 4
    chis = [
        (1.400, 1.500, 1.667),
        (1.222, 1.250, 1.286),
        (1.667, 2.000, 2.500),
    ]

    q_list, Q, w_list, defuzz_list, norm_defuzz = compute_fuzzy_swara(n, chis)

    print(f"\nSWARA difuso (n={n})\n")
    for i, q in enumerate(q_list, 1):
        print(f"q_{i} = {q}")
    print(f"\nQ = {Q}")

    print("\nPesos difusos normalizados:")
    for i, w in enumerate(w_list, 1):
        print(f"w_{i} = {w}")

    print("\nPesos defuzzificados (a+4b+c)/6:")
    for i, d in enumerate(defuzz_list, 1):
        print(f"defuzz(w_{i}) = {d:.12f}")

    print("\nPesos defuzzificados normalizados:")
    for i, nd in enumerate(norm_defuzz, 1):
        print(f"w_{i}^defuzz_norm = {nd:.12f}")

    print(f"\nSuma defuzzificados normalizados = {np.sum(norm_defuzz):.12f}")
