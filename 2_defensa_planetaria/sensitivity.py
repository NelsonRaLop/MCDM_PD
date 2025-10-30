import numpy as np
import matplotlib.pyplot as plt
import fuzzy_ranking
import asteroid_generator
import alternative_physics

def weighting(
    fuzzy_weights_fucom,
    fuzzy_weights_swara,
    decision_matrix_TFN,
    criteria_type,
    a,
    alts,
):
    """
    Versión minimalista:
    - asume alts está siempre presente (lista de nombres de alternativas)
    - decision_matrix_TFN se pasa tal cual a fuzzy_ranking.ranking_FTOPSIS
    - genera para cada criterio 2 escenarios: 1-a% y 1-2a%
    - ejecuta los escenarios para FUCOM y SWARA, plotea dos subplots y
      imprime resumen (victorias, mean_rank, rango)
    """

    alt_labels = list(alts)
    # determinar número de alternativas consultando ranking con pesos originales (una llamada rápida)
    sample_ranking = fuzzy_ranking.ranking_FTOPSIS(decision_matrix_TFN, criteria_type, fuzzy_weights_fucom)
    num_alternativas = len(sample_ranking)

    def _build_rankings(orig_weights):
        orig = np.array(orig_weights, dtype=float)   # shape (n_criteria, 3)
        n = orig.shape[0]
        rankings_list = []
        scenario_names = []
        for i in range(n):
            scenarios = [("dec_a", 1.0 - (a / 100.0)), ("dec_2a", 1.0 - (2.0 * a / 100.0))]
            for label, factor_change in scenarios:
                new_wi = orig[i] * factor_change
                new_weights = []
                for j in range(n):
                    if j == i:
                        new_weights.append(tuple(new_wi.tolist()))
                    else:
                        denom = 1.0 - orig[i]
                        numer = 1.0 - new_wi
                        with np.errstate(divide='ignore', invalid='ignore'):
                            factor = np.where(np.isclose(denom, 0.0), 1.0, numer / denom)
                        wj_orig = orig[j]
                        wj_new = wj_orig * factor
                        new_weights.append(tuple(wj_new.tolist()))
                scenario_id = f"C{i+1}_{label}"
                ranking = fuzzy_ranking.ranking_FTOPSIS(decision_matrix_TFN, criteria_type, new_weights)
                rankings_list.append(np.array(ranking))
                scenario_names.append(scenario_id)
        rankings_all = np.vstack(rankings_list)   # shape (2*n_criteria, num_alternativas)
        return rankings_all, scenario_names

    rankings_fucom, scenarios = _build_rankings(fuzzy_weights_fucom)
    rankings_swara, _ = _build_rankings(fuzzy_weights_swara)

    # PLOT: dos subplots, compartiendo eje x
    num_scenarios = rankings_fucom.shape[0]
    x = np.arange(1, num_scenarios + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    linestyles = ['-', '-', '-', '-']

    # FUCOM (arriba)
    for alt in range(num_alternativas):
        ax1.plot(x, rankings_fucom[:, alt],
                 marker=markers[alt % len(markers)],
                 linestyle=linestyles[alt % len(linestyles)],
                 label=alt_labels[alt])
    ax1.invert_yaxis()
    ax1.set_yticks(np.arange(1, num_alternativas + 1, 1))
    ax1.set_ylabel("Posición (1=mejor)")
    ax1.set_title("FUCOM - sensibilidad pesos")
    ax1.grid(True)
    ax1.legend(loc='upper right')

    # SWARA (abajo)
    for alt in range(num_alternativas):
        ax2.plot(x, rankings_swara[:, alt],
                 marker=markers[alt % len(markers)],
                 linestyle=linestyles[alt % len(linestyles)],
                 label=alt_labels[alt])
    ax2.invert_yaxis()
    ax2.set_yticks(np.arange(1, num_alternativas + 1, 1))
    ax2.set_xlabel("Escenarios")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.set_ylabel("Posición (1=mejor)")
    ax2.set_title("SWARA - sensibilidad pesos")
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Resumen: porcentaje de victorias, mean and range
    def _print_summary(rankings_all, label):
        ns = rankings_all.shape[0]
        print(f"\nSummary sensitivity_{label}:")
        for alt in range(num_alternativas):
            count_top = np.sum(rankings_all[:, alt] == 1)
            perc = 100.0 * count_top / ns
            print(f"  {alt_labels[alt]} ha sido 1ª en {perc:.1f}% de escenarios ({count_top}/{ns})")
        mean_ranks = rankings_all.mean(axis=0)
        min_ranks = rankings_all.min(axis=0)
        max_ranks = rankings_all.max(axis=0)
        print(f"  mean_rank = {np.round(mean_ranks, 4)}")
        print(f"  rank_range_min = {min_ranks}")
        print(f"  rank_range_max = {max_ranks}")

    _print_summary(rankings_fucom, "fucom")
    _print_summary(rankings_swara, "swara")

    return {
        'fucom': {'rankings': rankings_fucom, 'scenarios': scenarios},
        'swara': {'rankings': rankings_swara, 'scenarios': scenarios}
    }


def linguistic(aggregated_rank_fucom,
               expert_rankings_fucom,
               aggregated_fuzzy_weight_fucom,
               aggregated_rank_swara,
               expert_rankings_swara,
               aggregated_fuzzy_weight_swara,
               aggregated_decision_matrix_TFN,
               crits,
               alts,
               criteria_type,
               n,
               extra_sensitivity_output=None):
    """
    OAT sensitivity: compara FUCOM vs SWARA.
    Si extra_sensitivity_output (por ejemplo salida de sensitivity_cost_params) está
    presente, se añade como escenarios adicionales al plot (ej. 'cost').
    """

    rng = np.random.default_rng(12345)

    # índices (lanzará error si faltan)
    idx_tol = crits.index('tolerance')
    idx_int = crits.index('interest')
    idx_risk = crits.index('risk')

    n_alts = 4  # fijo según acordado

    # preparar expert_rankings arrays (ambos métodos)
    expert_rankings_fucom = np.asarray(expert_rankings_fucom)
    if expert_rankings_fucom.ndim == 1:
        expert_rankings_fucom = expert_rankings_fucom.reshape(1, -1)
    expert_rankings_swara = np.asarray(expert_rankings_swara)
    if expert_rankings_swara.ndim == 1:
        expert_rankings_swara = expert_rankings_swara.reshape(1, -1)

    # must have same n_exp for both (assumption)
    n_exp = expert_rankings_fucom.shape[0]

    # containers OAT: n x n_alts for each method and each crit
    results_OAT_fucom = {'tolerance': np.zeros((n, n_alts), dtype=float),
                         'interest':  np.zeros((n, n_alts), dtype=float),
                         'risk':      np.zeros((n, n_alts), dtype=float)}
    results_OAT_swara = {'tolerance': np.zeros((n, n_alts), dtype=float),
                         'interest':  np.zeros((n, n_alts), dtype=float),
                         'risk':      np.zeros((n, n_alts), dtype=float)}

    # ranking wrappers
    def compute_rank_fucom(matrix):
        return np.array(fuzzy_ranking.ranking_FTOPSIS(matrix, criteria_type, aggregated_fuzzy_weight_fucom), dtype=float)
    def compute_rank_swara(matrix):
        return np.array(fuzzy_ranking.ranking_FTOPSIS(matrix, criteria_type, aggregated_fuzzy_weight_swara), dtype=float)

    # OAT sampling: for each random TFN scenario compute both rankings
    for crit_name, col_idx in [('tolerance', idx_tol), ('interest', idx_int), ('risk', idx_risk)]:
        for s in range(n):
            M = [list(row) for row in aggregated_decision_matrix_TFN]
            for ai in range(n_alts):
                m = rng.uniform(0, 10.0)
                l = rng.uniform(0, m)
                u = rng.uniform(m, 10.0)
                M[ai][col_idx] = (float(l), float(m), float(u))
            results_OAT_fucom[crit_name][s, :] = compute_rank_fucom(M)
            results_OAT_swara[crit_name][s, :] = compute_rank_swara(M)

    # summaries: mean y percentiles (p_low/p_high) para cada método
    def make_summary(results_OAT):
        summary = {}
        for crit in ['tolerance', 'interest', 'risk']:
            arr = results_OAT[crit]   # n x n_alts
            summary[crit] = {
                'mean_rank': arr.mean(axis=0),
                'p_low': np.percentile(arr, 12.5, axis=0),   # o 5 si prefieres
                'p_high': np.percentile(arr, 87.5, axis=0),
                'all_ranks': arr
            }
        return summary

    summary_fucom = make_summary(results_OAT_fucom)
    summary_swara = make_summary(results_OAT_swara)

    # Si viene extra_sensitivity_output (por ejemplo out_cost), incorporarlo a los summaries
    extra_labels = []
    if extra_sensitivity_output is not None:
        esf = extra_sensitivity_output.get('summary_fucom', {})
        ess = extra_sensitivity_output.get('summary_swara', {})
        for key in esf.keys():
            extra_labels.append(key)
            e_fu = esf[key]
            e_sw = ess.get(key, {})

            def normalize_summary(e):
                if e is None:
                    return None
                mean = e.get('mean_rank') if 'mean_rank' in e else e.get('mean')
                p_low = e.get('p_low') if 'p_low' in e else None
                p_high = e.get('p_high') if 'p_high' in e else None
                allr = e.get('all_ranks') if 'all_ranks' in e else e.get('results')
                return {'mean_rank': mean, 'p_low': p_low, 'p_high': p_high, 'all_ranks': allr}

            norm_fu = normalize_summary(e_fu)
            norm_sw = normalize_summary(e_sw)
            if norm_fu is not None:
                summary_fucom[key] = norm_fu
            if norm_sw is not None:
                summary_swara[key] = norm_sw

        # --- PRINT SINTÉTICO de todos los summaries (cada criterio/escenario) ---
    fmt = lambda arr: np.array2string(np.asarray(arr), separator=' ', formatter={'float_kind':lambda x: f"{x:.4f}"})
    for crit in summary_fucom.keys():
        sfu = summary_fucom[crit]
        ssw = summary_swara.get(crit, None)
        if sfu is not None:
            mean_fu = sfu.get('mean_rank')
            p_low_fu = sfu.get('p_low')
            p_high_fu = sfu.get('p_high')
            if mean_fu is not None and p_low_fu is not None and p_high_fu is not None:
                print(f"summary_fucom['{crit}']['mean_rank'] = {fmt(mean_fu)}")
                print(f"summary_fucom['{crit}']['range p_low..p_high'] = {fmt(p_low_fu)} .. {fmt(p_high_fu)}")
        if ssw is not None:
            mean_sw = ssw.get('mean_rank')
            p_low_sw = ssw.get('p_low')
            p_high_sw = ssw.get('p_high')
            if mean_sw is not None and p_low_sw is not None and p_high_sw is not None:
                print(f"summary_swara['{crit}']['mean_rank'] = {fmt(mean_sw)}")
                print(f"summary_swara['{crit}']['range p_low..p_high'] = {fmt(p_low_sw)} .. {fmt(p_high_sw)}")

    
    # --- Plot: ahora crit_labels_oat puede tener los 3 básicos + extras (ej. cost) ---
    crit_labels_oat = ['tolerance', 'interest', 'risk'] + extra_labels

    # base positions for groups (aggregated, exp1..expK, *then* all scenarios in crit_labels_oat)
    base_positions = np.concatenate([[0], np.arange(1, 1 + n_exp), np.arange(1 + n_exp, 1 + n_exp + len(crit_labels_oat))])
    # left/right offset
    offset = 0.12
    left_pos = base_positions - offset
    right_pos = base_positions + offset

    # xticks labels placed at base_positions
    xticks_labels = ['aggregated'] + [f"exp{ei+1}" for ei in range(n_exp)] + crit_labels_oat
    xticks_pos = base_positions

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()

    # arrays of aggregated ranks
    agg_fu = np.asarray(aggregated_rank_fucom, dtype=float).reshape(-1)
    agg_sw = np.asarray(aggregated_rank_swara, dtype=float).reshape(-1)
    assert agg_fu.size == n_alts and agg_sw.size == n_alts, "aggregated_rank length mismatch"

    bar_halfwidth = 0.12
    line_thickness = 2.0

    # marker styles/colors:
    fu_band_color = 'C0'
    fu_marker = dict(marker='o', color='C0', markersize=6)
    sw_band_color = 'C2'
    sw_marker = dict(marker='o', color='C2', markersize=6)

    agg_marker_fu = dict(marker='o', color='C0', markersize=6)
    agg_marker_sw = dict(marker='o', color='C2', markersize=6)
    exp_marker_fu = dict(marker='o', color='C0', markersize=6)
    exp_marker_sw = dict(marker='o', color='C2', markersize=6)

    for ai in range(n_alts):
        ax = axes[ai]

        # 1) aggregated points (left FUCOM, right SWARA)
        ax.plot(left_pos[0], agg_fu[ai], **agg_marker_fu, zorder=8)
        ax.plot(right_pos[0], agg_sw[ai], **agg_marker_sw, zorder=8)

        # 2) expert points
        for ei in range(n_exp):
            ax.plot(left_pos[1 + ei], expert_rankings_fucom[ei, ai], **exp_marker_fu, zorder=7)
            ax.plot(right_pos[1 + ei], expert_rankings_swara[ei, ai], **exp_marker_sw, zorder=7)

        # 3) OAT groups + extra groups
        for xi, crit in enumerate(crit_labels_oat):
            gp_idx = 1 + n_exp + xi  # index in base_positions for this crit
            xx_fu = left_pos[gp_idx]
            xx_sw = right_pos[gp_idx]

            # obtener valores del summary; si no existen, saltar
            if crit not in summary_fucom or crit not in summary_swara:
                continue

            s_fu = summary_fucom[crit]
            s_sw = summary_swara[crit]

            # Helper para extraer ESCALAR para la alternativa ai de arrays en los summaries
            def get_scalar(summary_dict, key, idx):
                if summary_dict is None:
                    return None
                val = summary_dict.get(key)
                if val is None:
                    return None
                val_arr = np.asarray(val)
                # Si es array y tiene dimensión compatible, devolver elemento idx
                if val_arr.ndim == 1 and val_arr.size > idx:
                    return float(val_arr[idx])
                # Si ya es escalar
                if val_arr.shape == ():
                    return float(val_arr)
                # si no cumple, devolver None
                return None

            mean_fu = get_scalar(s_fu, 'mean_rank', ai)
            p_low_fu = get_scalar(s_fu, 'p_low', ai)
            p_high_fu = get_scalar(s_fu, 'p_high', ai)

            mean_sw = get_scalar(s_sw, 'mean_rank', ai)
            p_low_sw = get_scalar(s_sw, 'p_low', ai)
            p_high_sw = get_scalar(s_sw, 'p_high', ai)

            # si faltan datos, saltar este grupo
            if None in (mean_fu, p_low_fu, p_high_fu, mean_sw, p_low_sw, p_high_sw):
                continue

            # comparar escalares (ahora seguro)
            if np.isclose(p_low_fu, p_high_fu):
                ax.hlines(p_low_fu, xx_fu - bar_halfwidth, xx_fu + bar_halfwidth,
                          colors=fu_band_color, linewidth=line_thickness, zorder=3)
            else:
                ax.fill_between([xx_fu - bar_halfwidth, xx_fu + bar_halfwidth],
                                [p_low_fu, p_low_fu], [p_high_fu, p_high_fu],
                                color=fu_band_color, alpha=0.45, linewidth=0, zorder=2)
            ax.plot(xx_fu, mean_fu, **fu_marker, zorder=9)

            if np.isclose(p_low_sw, p_high_sw):
                ax.hlines(p_low_sw, xx_sw - bar_halfwidth, xx_sw + bar_halfwidth,
                          colors=sw_band_color, linewidth=line_thickness, zorder=3)
            else:
                ax.fill_between([xx_sw - bar_halfwidth, xx_sw + bar_halfwidth],
                                [p_low_sw, p_low_sw], [p_high_sw, p_high_sw],
                                color=sw_band_color, alpha=0.45, linewidth=0, zorder=2)
            ax.plot(xx_sw, mean_sw, **sw_marker, zorder=9)

        # formatting axis
        ax.set_ylim(0.8, n_alts + 0.2)
        ax.set_yticks(np.arange(1, n_alts + 1))
        ax.invert_yaxis()
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_title(alts[ai], fontsize=11)

        # set xticks on base positions and labels (centered between pairs)
        ax.set_xticks(base_positions)
        ax.set_xticklabels(xticks_labels, rotation=45, ha='right')

    # --- Leyenda sencilla, centrada debajo del gráfico ---
    fu_line = plt.Line2D([], [], color='C0', marker='o', linestyle='None', label='Mean F-FUCOM')
    sw_line = plt.Line2D([], [], color='C2', marker='o', linestyle='None', label='Mean F-SWARA')
    fu_patch = plt.Rectangle((0, 0), 1, 1, color='C0', alpha=0.45, label='Range F-FUCOM')
    sw_patch = plt.Rectangle((0, 0), 1, 1, color='C2', alpha=0.45, label='Range F-SWARA')

    fig.legend(handles=[fu_line, sw_line, fu_patch, sw_patch],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.show()

    out = {
        'results_OAT_fucom': results_OAT_fucom,
        'results_OAT_swara': results_OAT_swara,
        'summary_fucom': summary_fucom,
        'summary_swara': summary_swara,
        'fig': fig
    }
    return out

def cost_params(
    aggregated_fuzzy_weight_fucom,
    aggregated_fuzzy_weight_swara,
    aggregated_decision_matrix_TFN,
    crits,
    alts,
    criteria_type,
    n
):
    """
    Análisis de sensibilidad mínimo para 'cost'. Produce salidas compatibles con linguistic:
      - 'results_fucom' / 'results_swara' : n x n_alts
      - 'summary_fucom' / 'summary_swara' : dict con key 'cost' -> {'mean_rank','p5','p95','all_ranks'}
      - 'meta' : información útil
    """

    random_seed = 12345
    rng = np.random.default_rng(random_seed)

    # indice del criterio cost (fallará si no existe, intencional: mínimo)
    cost_idx = crits.index('cost')

    # asumimos n_alts = filas de la matriz TFN
    M_base = [list(row) for row in aggregated_decision_matrix_TFN]
    n_alts = len(M_base)
    # containers
    results_fucom = np.zeros((n, n_alts), dtype=float)
    results_swara = np.zeros((n, n_alts), dtype=float)

    # wrappers para ranking usando la implementación FTOPSIS existente
    def rank_fucom(matrix):
        return np.array(fuzzy_ranking.ranking_FTOPSIS(matrix, criteria_type, aggregated_fuzzy_weight_fucom), dtype=float)
    def rank_swara(matrix):
        return np.array(fuzzy_ranking.ranking_FTOPSIS(matrix, criteria_type, aggregated_fuzzy_weight_swara), dtype=float)

    # función compute_cost tal y como la tenías (devuelve TFN)
    def compute_cost(bus_new, instrument_complexity):
        # bus_new, instrument_complexity son TFNs (l,m,u); aplicar la fórmula elemento a elemento
        return tuple(np.exp(1.52 * np.array(bus_new) + 0.467 * np.array(instrument_complexity)))

    # loop de simulaciones
    for s in range(n):
        # copia base (lista de listas de TFNs)
        M = [list(row) for row in M_base]

        for ai in range(n_alts):
            # generar bus_new TFN según reglas corregidas
            b2 = rng.uniform(0.2, 1.0)
            b1 = rng.uniform(0.2, b2)
            b3 = rng.uniform(b2, 1.0)
            bus_tfn = (float(b1), float(b2), float(b3))

            # generar instrument_complexity TFN según reglas corregidas
            i2 = rng.uniform(0.0, 1.0)
            i1 = rng.uniform(0.0, i2)
            i3 = rng.uniform(i2, 1.0)
            inst_tfn = (float(i1), float(i2), float(i3))

            cost_tfn = compute_cost(bus_tfn, inst_tfn)

            # aseguramos tupla de floats (l,m,u)
            cost_tfn = (float(cost_tfn[0]), float(cost_tfn[1]), float(cost_tfn[2]))

            # insertar en la matriz simulada
            M[ai][cost_idx] = cost_tfn

        # calcular rankings y guardarlos
        results_fucom[s, :] = rank_fucom(M)
        results_swara[s, :] = rank_swara(M)

    # resumir: media y percentiles (hacemos p_low/p_high para compatibilidad con linguistic)
    p_low, p_high = (12.5, 87.5)
    mean_fu = results_fucom.mean(axis=0)
    p_low_fu = np.percentile(results_fucom, p_low, axis=0)
    p_high_fu = np.percentile(results_fucom, p_high, axis=0)

    mean_sw = results_swara.mean(axis=0)
    p_low_sw = np.percentile(results_swara, p_low, axis=0)
    p_high_sw = np.percentile(results_swara, p_high, axis=0)

    summary_fucom = {
        'cost': {
            'mean_rank': mean_fu,
            'p_low': p_low_fu,
            'p_high': p_high_fu,
            'all_ranks': results_fucom
        }
    }
    summary_swara = {
        'cost': {
            'mean_rank': mean_sw,
            'p_low': p_low_sw,
            'p_high': p_high_sw,
            'all_ranks': results_swara
        }
    }

    meta = {
        'n': n,
        'n_alts': n_alts,
        'cost_idx': cost_idx,
        'random_seed': random_seed
    }

    return {
        'results_fucom': results_fucom,
        'results_swara': results_swara,
        'summary_fucom': summary_fucom,
        'summary_swara': summary_swara,
        'meta': meta
    }

def ast_params(
    aggregated_fuzzy_weight_fucom,
    aggregated_fuzzy_weight_swara,
    aggregated_decision_matrix_TFN,
    crits,
    alts,
    criteria_type,
    n
):
    """Función mínima coherente con `linguistic` y `cost_params`.

    Comportamiento:
    - Ejecuta `n` simulaciones usando `asteroid_generator.asteroid_random()`.
    - Para cada simulación sustituye `test_mass` y `test_time` por los TFNs
      devueltos por `alternative_physics.test_mass` / `test_time`.
    - Calcula rankings con FUCOM y SWARA usando la firma existente
      `fuzzy_ranking.ranking_FTOPSIS(matrix, criteria_type, weights)`.
    - Devuelve estructura compatible con `linguistic` y `cost_params`.
    """

    # índices (fallará intencionalmente si no existen)
    idx_mass = crits.index("test_mass")
    idx_time = crits.index("test_time")

    # número de alternativas según la matriz
    M_base = [list(row) for row in aggregated_decision_matrix_TFN]
    n_alts = len(M_base)

    # contenedores de resultados (n simulaciones x n_alts)
    results_fucom = np.zeros((n, n_alts), dtype=float)
    results_swara = np.zeros((n, n_alts), dtype=float)

    # wrappers para llamar al ranking con la firma usada en el módulo
    def rank_fucom(matrix):
        return np.array(fuzzy_ranking.ranking_FTOPSIS(matrix, criteria_type, aggregated_fuzzy_weight_fucom), dtype=float)

    def rank_swara(matrix):
        return np.array(fuzzy_ranking.ranking_FTOPSIS(matrix, criteria_type, aggregated_fuzzy_weight_swara), dtype=float)

    # loop de simulaciones
    for s in range(n):
        # copia ligera de la matriz base
        M = [list(row) for row in M_base]

        for ai, alt in enumerate(alts):
            # muestreo de parámetros del asteroide
            params = asteroid_generator.asteroid_random()

            # cálculo de TFNs usando el módulo de física alternativo
            mass_tfn = alternative_physics.test_mass(alt, params=params)
            time_tfn = alternative_physics.test_time(alt, params=params)

            # insertar TFNs (se asume tupla inmutable (l,m,u))
            M[ai][idx_mass] = (float(mass_tfn[0]), float(mass_tfn[1]), float(mass_tfn[2]))
            M[ai][idx_time] = (float(time_tfn[0]), float(time_tfn[1]), float(time_tfn[2]))

        # calcular y almacenar rankings
        results_fucom[s, :] = rank_fucom(M)
        results_swara[s, :] = rank_swara(M)

    # resumir: media y percentiles 12.5/87.5 (compatibilidad con el resto)
    p_low, p_high = (12.5, 87.5)

    mean_fu = results_fucom.mean(axis=0)
    p_low_fu = np.percentile(results_fucom, p_low, axis=0)
    p_high_fu = np.percentile(results_fucom, p_high, axis=0)

    mean_sw = results_swara.mean(axis=0)
    p_low_sw = np.percentile(results_swara, p_low, axis=0)
    p_high_sw = np.percentile(results_swara, p_high, axis=0)

    # summaries con clave 'ast' para ser consistente con linguistic extra_labels
    summary_fucom = {
        'ast': {
            'mean_rank': mean_fu,
            'p_low': p_low_fu,
            'p_high': p_high_fu,
            'all_ranks': results_fucom
        }
    }

    summary_swara = {
        'ast': {
            'mean_rank': mean_sw,
            'p_low': p_low_sw,
            'p_high': p_high_sw,
            'all_ranks': results_swara
        }
    }

    # meta
    meta = {
        'n': n,
        'n_alts': n_alts,
        'idx_mass': idx_mass,
        'idx_time': idx_time,
        'sampler': 'asteroid_generator.asteroid_random'
    }

    # --- plotting: estilo similar a 'linguistic' ---
    x = np.arange(n_alts)
    dx = 0.12

    fig, ax = plt.subplots(figsize=(max(8, n_alts * 0.8), 6))

    # usar mismos colores y marcadores que linguistic
    fu_band_color = 'C0'
    sw_band_color = 'C2'
    fu_marker = dict(marker='o', color='C0', markersize=10)
    sw_marker = dict(marker='o', color='C2', markersize=10)

    # dibujar rangos (relleno) y medias como en linguistic
    for ai in range(n_alts):
        xx_fu = x[ai] - dx
        xx_sw = x[ai] + dx

        # FUCOM: rango
        if np.isclose(p_low_fu[ai], p_high_fu[ai]):
            ax.hlines(p_low_fu[ai], xx_fu - 0.12, xx_fu + 0.12, colors=fu_band_color, linewidth=2.0, zorder=2)
        else:
            ax.fill_between([xx_fu - 0.12, xx_fu + 0.12], [p_low_fu[ai], p_low_fu[ai]], [p_high_fu[ai], p_high_fu[ai]],
                            color=fu_band_color, alpha=0.45, linewidth=0, zorder=1)
        ax.plot(xx_fu, mean_fu[ai], **fu_marker, zorder=3)

        # SWARA: rango
        if np.isclose(p_low_sw[ai], p_high_sw[ai]):
            ax.hlines(p_low_sw[ai], xx_sw - 0.12, xx_sw + 0.12, colors=sw_band_color, linewidth=2.0, zorder=2)
        else:
            ax.fill_between([xx_sw - 0.12, xx_sw + 0.12], [p_low_sw[ai], p_low_sw[ai]], [p_high_sw[ai], p_high_sw[ai]],
                            color=sw_band_color, alpha=0.45, linewidth=0, zorder=1)
        ax.plot(xx_sw, mean_sw[ai], **sw_marker, zorder=3)

    # formato similar: eje y con solo 1..n_alts y legend debajo
    ax.set_xticks(x)
    ax.set_xticklabels(alts, rotation=45, ha='right')

    ax.set_xlabel('Alternativas')
    ax.set_ylabel('Ranking (posición)')
    ax.set_title(f'Análisis de sensibilidad asteroide (n={n})')

    ax.set_yticks(np.arange(1, n_alts + 1))
    ax.set_ylim(0.8, n_alts + 0.2)
    ax.invert_yaxis()
    ax.grid(axis='y', linestyle='--', alpha=0.25)

    # leyenda coherente con linguistic, centrada debajo
    fu_line = plt.Line2D([], [], color='C0', marker='o', linestyle='None', label='Mean F-FUCOM')
    sw_line = plt.Line2D([], [], color='C2', marker='o', linestyle='None', label='Mean F-SWARA')
    fu_patch = plt.Rectangle((0, 0), 1, 1, color='C0', alpha=0.45, label='Range F-FUCOM')
    sw_patch = plt.Rectangle((0, 0), 1, 1, color='C2', alpha=0.45, label='Range F-SWARA')

    fig.legend(handles=[fu_line, sw_line, fu_patch, sw_patch],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.show()

    return {
        'results_fucom': results_fucom,
        'results_swara': results_swara,
        'summary_fucom': summary_fucom,
        'summary_swara': summary_swara,
        'meta': meta,
        'fig': fig
    }
