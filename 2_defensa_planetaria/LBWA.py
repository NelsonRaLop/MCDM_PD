# Implemented as: https://www.researchgate.net/profile/Dragan-Pamucar/publication/335490951_New_model_for_determining_criteria_weights_Level_Based_Weight_Assessment_LBWA_model/links/5d68eb79a6fdccadeae45a72/New-model-for-determining-criteria-weights-Level-Based-Weight-Assessment-LBWA-model.pdf?origin=publication_detail&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uRG93bmxvYWQiLCJwYWdlIjoicHVibGljYXRpb25Eb3dubG9hZCIsInByZXZpb3VzUGFnZSI6InB1YmxpY2F0aW9uIn19&__cf_chl_tk=f4LaXxy4GTjivkQLyvR8RZ6G_Th2H52xkUh2UiRbr30-1758125703-1.0.1.1-FxaVFm2yFYUPuSQw8n2zAoyKjQhPdFvRvRCgsniDhu0
# Función que combina la clasificación en intervalos/subintervalos y el cálculo de pesos según tu especificación.
import numpy as np
import pandas as pd

def clasificar_y_calcular_pesos(valores, max_iter=1000000, verbose=True):
    valores = np.array(valores, dtype=float)
    n = len(valores)
    if n == 0:
        raise ValueError("La lista de valores está vacía.")

    # Paso 1: Referencia (máximo)
    ref = np.max(valores)

    # Paso 2: Crear intervalos descendentes [sup, inf) hasta cubrir todos los valores
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
            raise RuntimeError("Se alcanzó el máximo de iteraciones al construir intervalos; revisa los valores de entrada.")

    # Paso 3: Asignar valores a intervalos (almaceno índice y valor)
    asignaciones_por_intervalo = [[] for _ in range(len(intervalos))]
    for idx, v in enumerate(valores):
        for j, (sup, inf) in enumerate(intervalos):
            if inf < v <= sup:
                asignaciones_por_intervalo[j].append((idx, v))
                break

    # Paso 4: Número de subintervalos (n_subs = máximo número de elementos en un intervalo)
    n_subs = max((len(l) for l in asignaciones_por_intervalo), default=0) or 1
    r_elasticity = n_subs + 1

    # Paso 5: Determinar subintervalo de cada valor usando partición uniforme dentro del intervalo
    # Convención: sub_idx = 0 es el subintervalo superior (más cercano a sup)
    resultados = []  # lista de dicts con: idx, valor, intervalo(1-based), sub_idx (0-based)
    for idx, valor in enumerate(valores):
        for interval_idx, (sup, inf) in enumerate(intervalos):
            if inf < valor <= sup:
                intervalo_id = interval_idx + 1  # 1-based
                longitud = sup - inf
                tam_sub = longitud / n_subs if n_subs > 0 else longitud

                # calculamos sub_idx iterando como en tu algoritmo original
                found = False
                for sub_idx in range(n_subs):
                    sub_sup = sup - sub_idx * tam_sub
                    sub_inf = sub_sup - tam_sub
                    if sub_idx == 0:
                        if sub_inf <= valor <= sub_sup:
                            resultados.append({
                                "idx": idx,
                                "valor": valor,
                                "intervalo": intervalo_id,
                                "sub_idx": sub_idx
                            })
                            found = True
                            break
                    else:
                        if sub_inf < valor <= sub_sup:
                            resultados.append({
                                "idx": idx,
                                "valor": valor,
                                "intervalo": intervalo_id,
                                "sub_idx": sub_idx
                            })
                            found = True
                            break
                if not found:
                    # Por si hay redondeos y no se halló subintervalo, asignamos al último
                    resultados.append({
                        "idx": idx,
                        "valor": valor,
                        "intervalo": intervalo_id,
                        "sub_idx": n_subs - 1
                    })
                break

    # Paso 6: Calcular f_influ para cada valor
    # f_influ = r_elasticity / (numero_intervalo * r_elasticity + numero_subintervalo)
    f_influ = np.zeros(n, dtype=float)
    for r in resultados:
        idx = r["idx"]
        interval_num = r["intervalo"]
        sub_idx = r["sub_idx"]
        denom = interval_num * r_elasticity + sub_idx
        f_influ[idx] = r_elasticity / denom

    # Paso 7: Identificar el índice del "valor mayor" (si hay varios iguales al máximo se toma el primero)
    max_indices = np.where(valores == ref)[0]
    mayor_idx = int(max_indices[0])

    # Paso 8: Calcular w (peso del mayor)
    suma_otros = f_influ.sum() - f_influ[mayor_idx]
    w_mayor = 1.0 / (1.0 + suma_otros)

    # Paso 9: Pesos finales
    pesos = f_influ * w_mayor

    # Preparar tabla de salida ordenada por índice original (1-based para impresión)
    df = pd.DataFrame([
        {
            "Índice (1-based)": r["idx"] + 1,
            "Valor": r["valor"],
            "Intervalo": r["intervalo"],
            "Subintervalo": r["sub_idx"],
            "f_influ": f_influ[r["idx"]],
            "Peso": pesos[r["idx"]]
        }
        for r in resultados
    ]).sort_values("Índice (1-based)").reset_index(drop=True)

    # Si verbose, imprimimos en el formato solicitado
    if verbose:
        print(f"Referencia (máximo): {ref}")
        print(f"Número de intervalos: {len(intervalos)}")
        print(f"Número de subintervalos por intervalo (n_subs): {n_subs}")
        print(f"r_elasticity = n_subs + 1 = {r_elasticity}\n")

        for _, row in df.iterrows():
            idx1 = int(row["Índice (1-based)"])
            val = row["Valor"]
            intv = int(row["Intervalo"])
            sub = int(row["Subintervalo"])
            finf = row["f_influ"]
            peso = row["Peso"]
            mayor_mark = " (MAYOR)" if (idx1 - 1) == mayor_idx else ""
            print(f"Índice {idx1} — Valor {val:.4f} → Intervalo {intv}, Subintervalo {sub}{mayor_mark}; f_influ={finf:.6f}; Peso={peso:.6f}")

        print("\nResumen de pesos:")
        print(f"Índice del mayor (1-based): {mayor_idx + 1}")
        print(f"Peso del mayor: {w_mayor:.12f}")
        print(f"Suma de pesos (debería ser ~1): {pesos.sum():.12f}")

    return {
        "intervalos": intervalos,
        "n_subs": n_subs,
        "r_elasticity": r_elasticity,
        "f_influ": f_influ,
        "w_mayor": w_mayor,
        "pesos": pesos,
        "tabla": df
    }


# Ejemplo con tu lista
# valores = [0.12, 0.19, 0.11, 0.08, 0.15, 0.12, 0.14, 0.09]
# valores = [0.050, 0.125, 0.058, 0.107, 0.068, 0.047, 0.063, 0.15, 0.075, 0.094, 0.044, 0.042, 0.040, 0.038]
valores = np.array([0.1514130485567831,0.3136326448300416, 0.31363264483004155,0.1595951358510978,0.3302038133697736, 0.33282166497568644, 0.1154382696081658,0.2559638723163786, 0.25596387231637846, 0.0349469098582659,0.1100679377899250, 0.11357976686321081, 0.0352781851890795,0.0846676444537882, 0.10918685091166105])
# valores = 1/(valores_2)
print (valores)
resultado = clasificar_y_calcular_pesos(valores)

