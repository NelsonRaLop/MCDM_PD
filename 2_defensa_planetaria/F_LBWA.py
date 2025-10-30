# Script integrado: parte directamente de la lista de TFN y ejecuta todo el flujo.
import numpy as np
import pandas as pd

def defuzz(a, b, c):
    return (a + 4.0*b + c) / 6.0

def build_intervals_and_assign(valores, max_iter=1000000):
    valores = np.array(valores, dtype=float)
    n = len(valores)
    if n == 0:
        raise ValueError("Lista vacía")
    ref = np.max(valores)
    intervalos = []
    i = 1
    cubiertos = np.zeros(n, dtype=bool)
    while True:
        sup = ref / i
        inf = ref / (i + 1)
        intervalos.append((sup, inf))
        for idx, v in enumerate(valores):
            if inf < v <= sup:
                cubiertos[idx] = True
        if cubiertos.all():
            break
        i += 1
        if i > max_iter:
            raise RuntimeError("Máx iter alcanzado al construir intervalos")
    # asignar por intervalo
    asign_por_intervalo = [[] for _ in range(len(intervalos))]
    for idx, v in enumerate(valores):
        for j, (sup, inf) in enumerate(intervalos):
            if inf < v <= sup:
                asign_por_intervalo[j].append((idx, v))
                break
    n_subs = max((len(l) for l in asign_por_intervalo), default=0) or 1
    r_elasticity = n_subs + 1
    # determinar subintervalo de cada componente
    resultados = []
    for idx, v in enumerate(valores):
        for interval_idx, (sup, inf) in enumerate(intervalos):
            if inf < v <= sup:
                intervalo_id = interval_idx + 1  # 1-based
                longitud = sup - inf
                tam_sub = longitud / n_subs if n_subs > 0 else longitud
                found = False
                for sub_idx in range(n_subs):
                    sub_sup = sup - sub_idx * tam_sub
                    sub_inf = sub_sup - tam_sub
                    if sub_idx == 0:
                        if sub_inf <= v <= sub_sup:
                            resultados.append({
                                'idx': idx, 'valor': v,
                                'intervalo': intervalo_id, 'sub_idx': sub_idx
                            })
                            found = True
                            break
                    else:
                        if sub_inf < v <= sub_sup:
                            resultados.append({
                                'idx': idx, 'valor': v,
                                'intervalo': intervalo_id, 'sub_idx': sub_idx
                            })
                            found = True
                            break
                if not found:
                    resultados.append({
                        'idx': idx, 'valor': v,
                        'intervalo': intervalo_id, 'sub_idx': n_subs - 1
                    })
                break
    return intervalos, resultados, r_elasticity, n_subs, ref

def compute_weights_from_tfn_list(tfn_list, verbose=True):
    # Flatten components preserving order
    comps = []
    for t in tfn_list:
        if len(t) != 3:
            raise ValueError("Cada TFN debe ser (a,b,c)")
        comps.extend([float(t[0]), float(t[1]), float(t[2])])
    comps = np.array(comps, dtype=float)
    # Build intervals and assign subintervals
    intervalos, resultados, r_elasticity, n_subs, ref = build_intervals_and_assign(comps)
    # Compute f_influ per component
    n_comp = len(comps)
    f_influ_comp = np.zeros(n_comp, dtype=float)
    for r in resultados:
        idx = r['idx']
        interval_num = r['intervalo']
        sub_idx = r['sub_idx']
        denom = interval_num * r_elasticity + sub_idx
        f_influ_comp[idx] = r_elasticity / denom
    # Compute crisp weight of max component
    max_comp_idx = int(np.argmax(comps))  # index of component with max value
    f_max = f_influ_comp[max_comp_idx]
    sum_others = f_influ_comp.sum() - f_max
    w_max = 1.0 / (1.0 + sum_others)
    # Compute crisp weights per component
    weights_comp = np.zeros_like(f_influ_comp)
    for i, f in enumerate(f_influ_comp):
        if i == max_comp_idx:
            weights_comp[i] = w_max
        else:
            weights_comp[i] = f * w_max
    # Rebuild fuzzy weights per TFN (3 components each)
    n_tfn = len(tfn_list)
    weights_tfn_pre = [weights_comp[3*i:3*i+3].tolist() for i in range(n_tfn)]
    # Defuzz and normalize so sum(defuzz) == 1
    defuzz_vals_pre = [defuzz(*tri) for tri in weights_tfn_pre]
    sum_defuzz_pre = sum(defuzz_vals_pre)
    if sum_defuzz_pre <= 0:
        raise RuntimeError("Suma defuzz no positiva; revisar entradas")
    factor = 1.0 / sum_defuzz_pre
    weights_tfn_norm = [[c * factor for c in tri] for tri in weights_tfn_pre]
    defuzz_vals_norm = [defuzz(*tri) for tri in weights_tfn_norm]
    # Prepare a nice DataFrame for output
    rows = []
    for i, t in enumerate(tfn_list):
        rows.append({
            "TFN": f"T{i+1}",
            "TFN_tuple": t,
            "w_a_pre": weights_tfn_pre[i][0],
            "w_b_pre": weights_tfn_pre[i][1],
            "w_c_pre": weights_tfn_pre[i][2],
            "defuzz_pre": defuzz_vals_pre[i],
            "w_a_norm": weights_tfn_norm[i][0],
            "w_b_norm": weights_tfn_norm[i][1],
            "w_c_norm": weights_tfn_norm[i][2],
            "defuzz_norm": defuzz_vals_norm[i],
            "contains_max": (3*i <= max_comp_idx < 3*i+3)
        })
    df = pd.DataFrame(rows)
    # Verbose prints
    if verbose:
        print("=== Datos básicos ===")
        print(f"Referencia (máx) = {ref}\nNumber of intervals = {len(intervalos)}; n_subs = {n_subs}; r_elasticity = {r_elasticity}\n")
        print("Asignación de componentes (índice 1-based):")
        for r in resultados:
            idx1 = r['idx'] + 1
            print(f" Índice {idx1:2d}: valor={r['valor']:.6f} -> intervalo={r['intervalo']}, sub_idx={r['sub_idx']}; f_influ={f_influ_comp[r['idx']]:.12f}")
        print("\n-- Pesos crisp por componente --")
        print(f"Componente máximo (índice 1-based): {max_comp_idx+1}, f_max={f_max:.12f}, w_max (crisp)={w_max:.12f}")
        for i, (f, w) in enumerate(zip(f_influ_comp, weights_comp), start=1):
            mark = " (MAX)" if i-1 == max_comp_idx else ""
            print(f" {i:2d}: f_influ={f:.12f} -> peso_crisp={w:.12f}{mark}")
        print("\n-- Pesos difusos por TFN (ANTES de normalizar) y sus defuzz --")
        for i, row in df.iterrows():
            print(f" {row['TFN']}: {row['TFN_tuple']} -> pre = [{row['w_a_pre']:.12f}, {row['w_b_pre']:.12f}, {row['w_c_pre']:.12f}] ; defuzz_pre={row['defuzz_pre']:.12f} ; contains_max={row['contains_max']}")
        print(f"\nSuma defuzz (pre) = {sum_defuzz_pre:.12f}")
        print(f"Factor normalización = {factor:.12f}\n")
        print("-- Pesos difusos NORMALIZADOS por TFN y defuzz --")
        for i, row in df.iterrows():
            print(f" {row['TFN']}: {row['TFN_tuple']} -> norm = [{row['w_a_norm']:.12f}, {row['w_b_norm']:.12f}, {row['w_c_norm']:.12f}] ; defuzz_norm={row['defuzz_norm']:.12f}")
        print(f"\nSuma defuzz normalizada = {sum(defuzz_vals_norm):.12f}\n")
    return {
        "intervalos": intervalos,
        "resultados_componentes": resultados,
        "r_elasticity": r_elasticity,
        "n_subs": n_subs,
        "f_influ_comp": f_influ_comp,
        "max_comp_idx": max_comp_idx,
        "w_max": w_max,
        "weights_comp": weights_comp,
        "weights_tfn_pre": weights_tfn_pre,
        "weights_tfn_norm": weights_tfn_norm,
        "defuzz_pre": defuzz_vals_pre,
        "defuzz_norm": defuzz_vals_norm,
        "df": df
    }

# ---- funciones auxiliares para generar TFN a partir de chis ----
def divide_tfn_by_chi_reordered(tfn, chi):
    """
    Divide tfn (ta,tb,tc) por chi reordenado (chi.c, chi.b, chi.a):
      resultado = (ta / chi.c, tb / chi.b, tc / chi.a)
    Evita división por cero: lanza ValueError si algún componente de chi es 0.
    """
    ta, tb, tc = tfn
    ca, cb, cc = chi
    # denominador reordenado: (cc, cb, ca)
    if cc == 0 or cb == 0 or ca == 0:
        raise ValueError("Algún componente de chi es 0; división no definida.")
    return (ta / cc, tb / cb, tc / ca)


def build_tfn_from_chis_chain_reordered(n, chis):
    """
    Construye lista de n TFN con la regla:
      TFN1 = (1,1,1)
      TFN2 = TFN1 / chi1_reordered   (es decir / (chi1.c, chi1.b, chi1.a))
      TFN3 = TFN2 / chi2_reordered
      ...
    len(chis) debe ser n-1.
    Devuelve la lista de n TFN (tuplas float).
    """
    if len(chis) != n - 1:
        raise ValueError("len(chis) debe ser n-1")

    tfns = [(1.0, 1.0, 1.0)]
    # TFN2 = TFN1 ÷ chi1_reordered
    tfns.append(divide_tfn_by_chi_reordered(tfns[0], chis[0]))
    # subsecuentes
    for i in range(1, len(chis)):
        prev = tfns[-1]
        tfns.append(divide_tfn_by_chi_reordered(prev, chis[i]))
    return tfns

# Ejecutar ejemplo solicitado
if __name__ == "__main__":
    #TFN = [(0.67, 1.0, 1.5), (0.45, 1.0, 2.24), (1.67, 3.0, 5.2), (1.0, 1.3, 1.8)]
    # OP1 Dandole directamente los TFN
    # TFN = [(0.1514130485567831,0.3136326448300416, 0.31363264483004155),(0.1595951358510978,0.3302038133697736, 0.33282166497568644), (0.1154382696081658,0.2559638723163786, 0.25596387231637846), (0.0349469098582659,0.1100679377899250, 0.11357976686321081), (0.0352781851890795,0.0846676444537882, 0.10918685091166105)]
    # OP2 Dandole los chi
    n = 4
    chis = [
        #(0.67,1.00,1.50),   # chi^{(1)}
        #(1.00,2.00,3.73)
        (0.67,1.00,1.50),
        (1,1,1),
        (2.5, 3, 3.5)
    ]
    TFN = build_tfn_from_chis_chain_reordered(n, chis)
    print("TFN generados:")
    for i, t in enumerate(TFN, start=1):
        print(f"TFN{i} = ({t[0]:.12f}, {t[1]:.12f}, {t[2]:.12f})")

    # Calculo completo
    resultado = compute_weights_from_tfn_list(TFN, verbose=True)
