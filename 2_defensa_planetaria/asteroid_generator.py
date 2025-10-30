"""
Function: asteroid_sensitivity()
Returns a dict with:
  R_ast   : radius [m]
  P_ast   : rotation period [h]
  rho     : density [kg/m3]
  alb     : albedo (unitless)
  M_a_mol : molar mass [kg/mol]
  c_a     : parameter c_a
  k_t     : parameter k_t
  E_v     : sublimation enthalpy [J/kg]
  T_s     : sublimation temperature [K]
  type    : spectral type ('C','S','M')
  H       : absolute magnitude
  D_km    : diameter [km]

Notes:
 - Uses numpy for math and random sampling to keep code concise and fast.
 - Sampling follows the specification provided: power-law for size, categorical
   type, conditional uniforms, deterministic H, truncated normal for P_ast
   (95% CI -> sigma = 0.09/1.96), and mixtures for overlapping intervals.

"""
import numpy as np

# ---------- constants and configuration -------------------------------------
ALPHA = 2.354  # exponent in N(>D) = 942 * D^{-ALPHA}
R_MIN = 50.0   # meters
R_MAX = 150.0  # meters

TYPE_PROBS = [('C', 0.75), ('S', 0.17), ('M', 0.08)]

# type-dependent ranges for alb, M_a_mol (kg/mol) and rho (kg/m3)
TYPE_RANGES = {
    'C': {
        'alb': (0.03, 0.09),
        'M_a_mol': (60e-3, 95e-3),
        'rho': ((1.38 - 0.02) * 1e3, (1.38 + 0.02) * 1e3),
    },
    'S': {
        'alb': (0.10, 0.22),
        'M_a_mol': (95e-3, 140e-3),
        'rho': ((2.71 - 0.02) * 1e3, (2.71 + 0.02) * 1e3),
    },
    'M': {
        'alb': (0.10, 0.18),
        'M_a_mol': (55e-3, 59e-3),
        'rho': ((5.32 - 0.07) * 1e3, (5.32 + 0.07) * 1e3),
    }
}

# mixtures for overlapping-interval parameters: list of (prob, (a,b))
C_A_MIX = [
    (0.1,   (375.0, 470.0)),
    (0.3667,(470.0, 600.0)),
    (0.333, (470.0, 750.0)),
    (0.2,   (600.0, 750.0)),
]

K_T_MIX = [
    (0.1, (0.2, 0.5)),
    (0.4, (1.47, 1.6)),
    (0.5, (0.2, 2.0)),
]

E_V_MIX = [
    (0.0667, (2.7e5, 1.0e6)),
    (0.3333, (2.7e5, 6.0e6)),
    (0.2333, (4.0e6, 6.0e6)),
    (0.3667, (1.0e7, 1.9686e7)),
]

T_S_MIX = [
    (1.0/3.0, (1700.0, 1720.0)),
    (1.0/3.0, (1720.0, 1812.0)),
    (1.0/3.0, (1700.0, 1812.0)),
]

# ----------------------------------------------------------------------------

def _choose_from_probs(items):
    """Choose one item from a list of (value, prob) pairs.

    Returns the selected value.
    """
    values, probs = zip(*items)
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()        # normalizar por seguridad
    idx = np.random.choice(len(values), p=probs)
    return values[idx]


def _sample_uniform_range(rng):
    """Uniform sample within range tuple (a,b)."""
    a, b = rng
    return np.random.uniform(a, b)


def _sample_mixture(mix):
    """Select an interval according to probabilities then sample uniformly inside it.

    mix: list of (prob, (a,b))
    """
    probs, ranges = [], []
    for p, r in mix:
        probs.append(p)
        ranges.append(r)
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()        # normalizar por seguridad
    idx = np.random.choice(len(ranges), p=probs)
    a, b = ranges[idx]
    return np.random.uniform(a, b)


def _sample_powerlaw_R(R_min=R_MIN, R_max=R_MAX, alpha=ALPHA):
    """Sample radius R (m) from power-law derived from N(>D) = K * D^{-alpha}.

    Steps made explicit and simple:
      - differential PDF in R is proportional to R^{-(alpha+1)}
      - use inverse CDF sampling for power-law
    """
    k = alpha + 1.0  # exponent in PDF: R^{-k}

    # compute constants for inverse CDF (work with simple intermediate variables)
    exponent = 1.0 - k  # this is negative
    Rmin_term = R_min ** exponent
    Rmax_term = R_max ** exponent

    # sample uniform in [0,1]
    u = np.random.random()

    # invert CDF step-by-step for readability
    inner = u * (Rmax_term - Rmin_term) + Rmin_term
    R = inner ** (1.0 / exponent)
    return R


def asteroid_random():
    """Generate one asteroid parameter sample (simple and readable).

    Returns a dict with named keys.
    """
    # 1) Radius sampled from power-law (meters)
    R_ast = _sample_powerlaw_R()

    # 2) Spectral type (C, S, M)
    type_sel = _choose_from_probs(TYPE_PROBS)

    # 3) albedo, molar mass and density depending on type
    alb = _sample_uniform_range(TYPE_RANGES[type_sel]['alb'])
    M_a_mol = _sample_uniform_range(TYPE_RANGES[type_sel]['M_a_mol'])
    rho = _sample_uniform_range(TYPE_RANGES[type_sel]['rho'])

    # 4) diameter (km) and absolute magnitude H (deterministic)
    D_km = R_ast / 500.0  # because D[km] = 2*R[m]/1000 = R/500
    # H from given formula, no added noise
    H = (3.1236 - 0.5 * np.log10(alb) - np.log10(D_km)) / 0.2

    # 5) rotation period: Gaussian with sigma from 95% CI, truncated to positive

    mu_ln = -0.5091 * H + 11.32
    sigma_ln = 3.32 / 1.96  # ≈ 1.693877551

    P_ast = np.random.lognormal(mean=mu_ln, sigma=sigma_ln)
    if P_ast <0 :
        P_ast = 0



    # 6) overlapping-interval parameters (select interval then uniform)
    c_a = _sample_mixture(C_A_MIX)
    k_t = _sample_mixture(K_T_MIX)
    E_v = _sample_mixture(E_V_MIX)
    T_s = _sample_mixture(T_S_MIX)

    return {
        'R_ast': R_ast,
        'P_ast': P_ast,
        'rho': rho,
        'alb': alb,
        'M_a_mol': M_a_mol,
        'c_a': c_a,
        'k_t': k_t,
        'E_v': E_v,
        'T_s': T_s,
    }
