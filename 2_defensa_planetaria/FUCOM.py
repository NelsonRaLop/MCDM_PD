# Implemented as https://pdfs.semanticscholar.org/809b/d4cc55c66036264b463234d9e582471e9d31.pdf

# Archivo: minimizar_eta_general.py
import numpy as np
from scipy.optimize import linprog

def resolver_min_eta(n, chis):
    """
    Resuelve el problema para n conjuntos (a_i,b_i,c_i), i=1..n,
    y chis: lista de n-1 tuplas (chi_j1, chi_j2, chi_j3).
    Devuelve (x_dict, eta, res) donde x_dict contiene arrays a,b,c.
    """
    if n < 2:
        raise ValueError("n debe ser >= 2")
    if len(chis) != n-1:
        raise ValueError("chis debe tener longitud n-1")
    for ch in chis:
        if len(ch) != 3:
            raise ValueError("cada chi debe ser long. 3 (chi1,chi2,chi3)")

    # Variables en vector x: [a1,b1,c1, a2,b2,c2, ..., a_n,b_n,c_n, eta]
    nv = 3*n + 1
    idx = lambda i, var: 3*(i) + {"a":0,"b":1,"c":2}[var]  # i desde 0..n-1
    idx_eta = 3*n

    # Objetivo: minimizar eta
    cobj = np.zeros(nv)
    cobj[idx_eta] = 1.0

    A_ub = []
    b_ub = []

    def add_abs_constraint(ix, k, iy):
        # ix - k*iy - eta <= 0
        row1 = np.zeros(nv); row1[ix] = 1; row1[iy] = -k; row1[idx_eta] = -1
        # -ix + k*iy - eta <= 0
        row2 = np.zeros(nv); row2[ix] = -1; row2[iy] = k; row2[idx_eta] = -1
        A_ub.append(row1); b_ub.append(0.0)
        A_ub.append(row2); b_ub.append(0.0)

    # Construir restricciones:
    # Para j = 0..n-2 (que corresponde a j=1..n-1 en notación del enunciado)
    for j in range(n-1):
        chi_j1, chi_j2, chi_j3 = chis[j]
        # a_j vs c_{j+1}
        add_abs_constraint(idx(j,"a"), chi_j1, idx(j+1,"c"))
        # b_j vs b_{j+1}
        add_abs_constraint(idx(j,"b"), chi_j2, idx(j+1,"b"))
        # c_j vs a_{j+1}
        add_abs_constraint(idx(j,"c"), chi_j3, idx(j+1,"a"))

    # Comparación con el "siguiente del siguiente" para j=0..n-3
    for j in range(n-2):
        # producto de chi_j * chi_{j+1} por componente
        chi_j = chis[j]
        chi_jp1 = chis[j+1]
        # a_j vs c_{j+2} con chi_j1 * chi_{j+1,1}
        add_abs_constraint(idx(j,"a"), chi_j[0]*chi_jp1[0], idx(j+2,"c"))
        # b_j vs b_{j+2} con chi_j2 * chi_{j+1,2}
        add_abs_constraint(idx(j,"b"), chi_j[1]*chi_jp1[1], idx(j+2,"b"))
        # c_j vs a_{j+2} con chi_j3 * chi_{j+1,3}
        add_abs_constraint(idx(j,"c"), chi_j[2]*chi_jp1[2], idx(j+2,"a"))

    # Orden a_i <= b_i <= c_i  =>  a_i - b_i <= 0 , b_i - c_i <= 0
    for i in range(n):
        row = np.zeros(nv); row[idx(i,"a")] = 1; row[idx(i,"b")] = -1
        A_ub.append(row); b_ub.append(0.0)
        row2 = np.zeros(nv); row2[idx(i,"b")] = 1; row2[idx(i,"c")] = -1
        A_ub.append(row2); b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Igualdad suma: sum_i (a_i + 4*b_i + c_i) = 6
    A_eq = np.zeros((1, nv))
    for i in range(n):
        A_eq[0, idx(i,"a")] = 1.0
        A_eq[0, idx(i,"b")] = 4.0
        A_eq[0, idx(i,"c")] = 1.0
    b_eq = np.array([6.0])

    # Bounds: a_i >= 0 para todos i, eta >= 0. b_i,c_i libres (None)
    bounds = []
    for i in range(n):
        bounds.append((0.0, None))   # a_i >= 0
        bounds.append((None, None))  # b_i libre
        bounds.append((None, None))  # c_i libre
    bounds.append((0.0, None))      # eta >= 0

    # Llamada al solver
    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return None, None, res

    x = res.x
    a = np.array([x[idx(i,"a")] for i in range(n)])
    b = np.array([x[idx(i,"b")] for i in range(n)])
    c = np.array([x[idx(i,"c")] for i in range(n)])
    eta_val = max(x[idx_eta], 0.0)  # por si hay tiny num negativo

    x_dict = {"a": a, "b": b, "c": c}
    return x_dict, eta_val, res

# Ejemplo de uso:
if __name__ == "__main__":
    # ejemplo con n=4 y valores chi (puedes sustituir por tus chi reales)
    n = 6
    chis = [
    (0.67,1,1.5),
    (1,2,3.73),
    (1,1.5,2.33),
    (1,1.33,1.8),
    (0.78,1,1.29) 
    ]
    
    x_dict, eta_val, res = resolver_min_eta(n, chis)
    if res.success:
        print("eta mínimo =", eta_val)
        for i in range(n):
            print(f"({i+1}) a{i+1},b{i+1},c{i+1} =",
                  (x_dict["a"][i], x_dict["b"][i], x_dict["c"][i]))
    else:
        print("Solver falló:", res.message)
