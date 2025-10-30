import numpy as np

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
        "VH": [7, 9, 10],
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

    return aggregated

## Evaluation 

# WRT Technological maturity
KI_maturity = expert_agregation(("H","H","VH","H","VH","H","H","M","VH","VH"))
print("Agregated KI_maturity:", KI_maturity)
IBD_maturity = expert_agregation(("M","LO","M","VL","H","H","VL","LO","VH","M"))
print("Agregated IDB_maturity:", IBD_maturity)
EGT_maturity = expert_agregation(("VL","M","LO","LO","VL","LO","LO","LO","VH","LO"))
print("Agregated EGT_maturity:", EGT_maturity)
LA_maturity = expert_agregation(("VL","LO","VL","VL","VL","LO","VL","VL","VH","LO"))
print("Agregated LA_maturity:", LA_maturity)



# WRT Operation Risk
KI_risk = expert_agregation(("M","LO","VH","LO","H","M","H","M","H","M"))
print("Agregated KI_risk:", KI_risk)
IDB_risk = expert_agregation(("VL","H","M","VH","M","LO","VH","H","H","LO"))
print("Agregated IDB_risk:", IDB_risk)
EGT_risk = expert_agregation(("VH","M","M","H","VH","LO","H","H","H","LO"))
print("Agregated EGT_risk:", EGT_risk)
LA_risk = expert_agregation(("H","M","LO","VH","M","H","VH","H","H","LO"))
print("Agregated LA_risk:", LA_risk)




