import numpy as np
import pandas as pd
import fuzzy_ranking

def expert_agregation(opinions):
    """
    Aggregates expert opinions expressed as linguistic variables (VL,L,M,H,VH)
    into a single fuzzy triangular number [a, b, c].

    Parameters:
        opinions (tuple): Tuple of linguistic labels, e.g. ("H", "L", "M")

    Returns:
        list: Aggregated triangular fuzzy number [a, b, c]
    """

    # Linguistic labels
    linguistic_to_triangular = {
        "VL": [0, 1, 3],
        "LO": [1, 3, 5],
        "M":  [3, 5, 7],
        "H":  [5, 7, 9],
        "VH": [7, 9, 10]
    }

    # To store the values
    triangular_values = []

    for op in opinions:
        if op not in linguistic_to_triangular:
            raise ValueError(f"Etiqueta desconocida: {op}")
        triangular_values.append(linguistic_to_triangular[op])

    # TO numpy to perform vector operations
    triangular_array = np.array(triangular_values)

    # Column mean: [mean(a), mean(b), mean(c)]
    aggregated = np.mean(triangular_array, axis=0)
    #print(aggregated)

    # List if neccesary
    #opinion_agregated = aggregated.tolist()

    return aggregated.tolist()


def decision_matrix(performance):
    '''
    This functions generates the decision matrix given the complete information
    of the candidates 

    return  decision_matrix_TFN (crisp numbers into TFN)
    '''

    decision_matrix_TFN = []

    for index, row in df.iterrows():
        row_tfn = []
        for val in row:
            if isinstance(val, (int, float)):  # Valor crisp
                row_tfn.append([val, val, val])
            elif isinstance(val, list) and len(val) == 3:
                row_tfn.append(val)
            else:
                raise ValueError(f"Formato no reconocido para el valor: {val}")
        decision_matrix_TFN.append(tuple(row_tfn))  # Puedes usar lista si prefieres

    return decision_matrix_TFN


################ MAIN ###################
# RAW data

## Evaluation 

# WRT Interest
KI_interest = expert_agregation(("M",))
IBD_interest = expert_agregation(("VH",))
EGT_interest = expert_agregation(("H",))
LA_interest = expert_agregation(("M",))

# WRT Tolerance
KI_tolerance = expert_agregation(("H",))
IBD_tolerance = expert_agregation(("M",))
EGT_tolerance = expert_agregation(("M",))
LA_tolerance = expert_agregation(("LO",))

# WRT Technological maturity
KI_maturity = expert_agregation(("H","H","VH","H","VH","H","H","M","VH","VH"))
IBD_maturity = expert_agregation(("M","LO","M","VL","H","H","VL","LO","VH","M"))
EGT_maturity = expert_agregation(("VL","M","LO","LO","VL","LO","LO","LO","VH","LO"))
LA_maturity = expert_agregation(("VL","LO","VL","VL","VL","LO","VL","VL","VH","LO"))

# WRT Operation Risk
KI_risk = expert_agregation(("M","LO","VH","LO","H","M","H","M","H","M"))
IDB_risk = expert_agregation(("VL","H","M","VH","M","LO","VH","H","H","LO"))
EGT_risk = expert_agregation(("VH","M","M","H","VH","LO","H","H","H","LO"))
LA_risk = expert_agregation(("H","M","LO","VH","M","H","VH","H","H","LO"))

# PD Payload

data = {
    "Payload":   ["KI", "IBD", "EGT","LA"],
    "Interest":  [KI_interest, IBD_interest, EGT_interest, LA_interest],  # From experts TFN
    "Time_test": [15, 5.24, 24.77, 2.4], # Crisp computed [days]
    "Mass_test": [51, 80.2, 32.83, 255], # Crisp computed [kg]
    "Tolerance": [KI_tolerance, IBD_tolerance, EGT_tolerance, LA_tolerance],  # From experts TFN
    "Maturity":  [KI_maturity, IBD_maturity, EGT_maturity, LA_maturity],  # From experts TFN
    "Risk":      [KI_risk, IDB_risk, EGT_risk, LA_risk]  # From experts TFN
}
df = pd.DataFrame(data) 
df.set_index("Payload", inplace=True)

# Decision matrix
decision_matrix_TFN = decision_matrix(df)

# Description of criteria
criterion_type= ['max','min','min','max','max','min']

# Generation of weights
fuzzy_weights = [ ( 0.166667, 0.166667 , 0.166667 )  , 
                  ( 0.166667 , 0.166667 , 0.166667 ) , 
                  ( 0.166667 , 0.166667 , 0.166667 ) , 
                  ( 0.166667 , 0.166667 , 0.166667 ) , 
                  ( 0.166667 , 0.166667 , 0.166667 ) , 
                  ( 0.166667 , 0.166667 , 0.166667 ) ]

# Generation of rankings
fuzzy_ranking.ranking_FTOPSIS(decision_matrix_TFN,criterion_type,fuzzy_weights)
fuzzy_ranking.ranking_FWASPAS(decision_matrix_TFN,criterion_type,fuzzy_weights)
fuzzy_ranking.ranking_FMARCOS(decision_matrix_TFN,criterion_type,fuzzy_weights)
fuzzy_ranking.ranking_FVIKOR(decision_matrix_TFN,criterion_type,fuzzy_weights)
fuzzy_ranking.ranking_FCOCOSO(decision_matrix_TFN,criterion_type,fuzzy_weights)