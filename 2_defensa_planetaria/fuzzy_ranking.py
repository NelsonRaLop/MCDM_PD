import numpy as np

def ranking_FTOPSIS(decision_matrix_TFN,criteria_type,fuzzy_weights):
    '''
    This function computes the ranking using fuzzy TOPSIS as described in:

    https://www.sciencedirect.com/science/article/pii/S187705091631273X
    https://arxiv.org/pdf/1205.5098
    '''
    # Initialization
    fuzzy_weights = np.array(fuzzy_weights)
    m = len(decision_matrix_TFN[0]) # Number of criteria
    n = len(decision_matrix_TFN) # Number of alternatives
    decision_matrix_TFN = np.array(decision_matrix_TFN)
    
    w_a = np.zeros((m))
    w_b = np.zeros((m))
    w_c = np.zeros((m))

    a = np.zeros((n,m))
    b = np.zeros((n,m))
    c = np.zeros((n,m))
    a_norm = np.zeros((n,m))
    b_norm = np.zeros((n,m))
    c_norm = np.zeros((n,m))
    v_weighted_norm_a = np.zeros((n,m))
    v_weighted_norm_b = np.zeros((n,m))
    v_weighted_norm_c = np.zeros((n,m))

    FPIS_a = np.zeros ((m))
    FPIS_b = np.zeros ((m))
    FPIS_c = np.zeros ((m))

    FNIS_a = np.zeros ((m))
    FNIS_b = np.zeros ((m))
    FNIS_c = np.zeros ((m))

    d_PIS = np.zeros ((n,m))
    d_NIS = np.zeros ((n,m))

    d_PIS_tot = np.zeros ((n))
    d_NIS_tot = np.zeros ((n))

    CC = np.zeros((n))

    # Get triangular numbers
    for j in range(0, m):

        # Weights
        w_a[j] = fuzzy_weights [j,0]
        w_b[j] = fuzzy_weights [j,1]
        w_c[j] = fuzzy_weights [j,2]

        # Performance
        for i in range(0, n):
            a[i,j], b[i,j], c[i,j]  = decision_matrix_TFN[i,j]

    # Normalization
    min_columns_a = np.argmin(a, axis=0)
    max_columns_c = np.argmax(c, axis=0)

    for j in range(0, m):
        for i in range(0, n):

            a_min = a[min_columns_a[j],j]
            c_max = c[max_columns_c[j],j]
            
            if (criteria_type[j] == 'min'): # Normalization for cost criteria
                a_norm[i,j] = (a_min)/(c[i,j])
                b_norm[i,j] = (a_min)/(b[i,j])
                c_norm[i,j] = (a_min)/(a[i,j])
            
            if (criteria_type[j] == 'max'): # Normalization for benefit criteria
                a_norm[i,j] = (a[i,j])/(c_max)
                b_norm[i,j] = (b[i,j])/(c_max)
                c_norm[i,j] = (c[i,j])/(c_max)    

    # Weighted normalized decision matrix
    for i in range(0,n):
        for j in range(0,m):
            v_weighted_norm_a[i,j] = a_norm[i,j]*w_a[j]
            v_weighted_norm_b[i,j] = b_norm[i,j]*w_b[j]
            v_weighted_norm_c[i,j] = c_norm[i,j]*w_c[j]

    max_columns_v = np.argmax(v_weighted_norm_c, axis=0)
    min_columns_v = np.argmin(v_weighted_norm_a, axis=0)
   
    #Fuzzy positive ideal and negative ideal
    for j in range(0,m):

        FPIS_a[j] =  v_weighted_norm_c[max_columns_v[j],j]
        FPIS_b[j] =  v_weighted_norm_c[max_columns_v[j],j]
        FPIS_c[j] =  v_weighted_norm_c[max_columns_v[j],j]
  
        FNIS_a[j] =  v_weighted_norm_a[min_columns_v[j],j]                                                                                                                                                                                                                                
        FNIS_b[j] =  v_weighted_norm_a[min_columns_v[j],j]                                                                                                                                                                                                                                
        FNIS_c[j] =  v_weighted_norm_a[min_columns_v[j],j]                                                                                                                                                                                                                                
  
    #Compute the distances of each alternative to FPIS and FNIS
    for i in range(0,n):
        for j in range(0,m):
            d_PIS[i,j] = np.sqrt(1/3*((v_weighted_norm_a[i,j]-FPIS_a[j])**2+(v_weighted_norm_b[i,j]-FPIS_b[j])**2+(v_weighted_norm_c[i,j]-FPIS_c[j])**2))
            d_NIS[i,j] = np.sqrt(1/3*((v_weighted_norm_a[i,j]-FNIS_a[j])**2+(v_weighted_norm_b[i,j]-FNIS_b[j])**2+(v_weighted_norm_c[i,j]-FNIS_c[j])**2))

        d_PIS_tot[i] = np.sum(d_PIS[i,:])
        d_NIS_tot[i] = np.sum(d_NIS[i,:])
        # Compute the closeness coefficient
        CC[i] = d_NIS_tot[i] /(d_NIS_tot[i] + d_PIS_tot[i])

    #print('FTOPSIS performance', CC)
    positions = np.argsort(np.argsort(-CC)) + 1  # 1 = mejor alternativa

    return positions

def ranking_FWASPAS(decision_matrix_TFN,criteria_type,fuzzy_weights):
    '''
    This function computes the ranking using fuzzy WSM WPM ans WASPAS as described in:

    https://link.springer.com/article/10.1007/s13369-022-07127-3

    BETTER:

    https://www.mdpi.com/2071-1050/11/2/424

    To the power of something:
    https://www.mdpi.com/2071-1050/14/9/4972
    '''
    # Initialization
    fuzzy_weights = np.array(fuzzy_weights)
    m = len(decision_matrix_TFN[0]) # Number of criteria
    n = len(decision_matrix_TFN) # Number of alternatives
    decision_matrix_TFN = np.array(decision_matrix_TFN)
    
    w_a = np.zeros((m))
    w_b = np.zeros((m))
    w_c = np.zeros((m))

    a = np.zeros((n,m))
    b = np.zeros((n,m))
    c = np.zeros((n,m))
    a_norm = np.zeros((n,m))
    b_norm = np.zeros((n,m))
    c_norm = np.zeros((n,m))

    Q_a_WSM = np.zeros((n))
    Q_b_WSM = np.zeros((n))
    Q_c_WSM = np.zeros((n))
    performance_WSM = np.zeros((n))

    Q_a_WPM = np.zeros((n))
    Q_b_WPM = np.zeros((n))
    Q_c_WPM = np.zeros((n))
    performance_WPM = np.zeros((n))

    performance_WASPAS = np.zeros((n))

    # Get triangular numbers
    for j in range(0, m):

        # Weights
        w_a[j] = fuzzy_weights [j,0]
        w_b[j] = fuzzy_weights [j,1]
        w_c[j] = fuzzy_weights [j,2]

        # Performance
        for i in range(0, n):
            a[i,j], b[i,j], c[i,j]  = decision_matrix_TFN[i,j]

    # Normalization
    min_columns_a = np.argmin(a, axis=0)
    max_columns_c = np.argmax(c, axis=0)

    for j in range(0, m):
        for i in range(0, n):

            a_min = a[min_columns_a[j],j]
            c_max = c[max_columns_c[j],j]
            
            if (criteria_type[j] == 'min'): # Normalization for cost criteria
                a_norm[i,j] = (a_min)/(c[i,j])
                b_norm[i,j] = (a_min)/(b[i,j])
                c_norm[i,j] = (a_min)/(a[i,j])
            
            if (criteria_type[j] == 'max'): # Normalization for benefit criteria
                a_norm[i,j] = (a[i,j])/(c_max)
                b_norm[i,j] = (b[i,j])/(c_max)
                c_norm[i,j] = (c[i,j])/(c_max)    
    
    for i in range(0,n): # WPM WSM
        Q_a_WSM[i] = np.sum(a_norm[i,:] * w_a)
        Q_b_WSM[i] = np.sum(b_norm[i,:] * w_b)
        Q_c_WSM[i] = np.sum(c_norm[i,:] * w_c)
        performance_WSM[i] = (Q_a_WSM[i] + 4*Q_b_WSM[i] + Q_c_WSM[i])/6 

        Q_a_WPM[i] = np.prod(a_norm[i,:] ** w_c)
        Q_b_WPM[i] = np.prod(b_norm[i,:] ** w_b)
        Q_c_WPM[i] = np.prod(c_norm[i,:] **w_a)
        performance_WPM[i] = (Q_a_WPM[i] + 4*Q_b_WPM[i] + Q_c_WPM[i])/6 

    # WASPAS

    # Compute Lambda
    lambda_WASPAS = np.sum(performance_WPM) / (np.sum(performance_WPM)+np.sum(performance_WSM))

    # Compute WASPAS performance
    for i in range(0,n):
        performance_WASPAS[i] = lambda_WASPAS * performance_WSM[i] + (1 - lambda_WASPAS)*performance_WPM[i]

    # print('WSM', performance_WSM)
    # print('WPM',performance_WPM)
    print('FWASPAS performance',performance_WASPAS)

    return 

def ranking_FMARCOS(decision_matrix_TFN,criteria_type,fuzzy_weights):
    '''
    This function computes the ranking using fuzzy MARCOS as described in:

    https://www.mdpi.com/2227-7390/8/3/457

    Ideal and antiideal:
    https://www.researchgate.net/publication/359361794_DIBR_-_Fuzzy_MARCOS_model_for_selecting_a_location_for_a_heavy_mechanized_bridge
    '''
    # Initialization
    fuzzy_weights = np.array(fuzzy_weights)
    m = len(decision_matrix_TFN[0]) # Number of criteria
    n = len(decision_matrix_TFN) # Number of alternatives
    decision_matrix_TFN = np.array(decision_matrix_TFN)
    
    w_a = np.zeros((m))
    w_b = np.zeros((m))
    w_c = np.zeros((m))

    a = np.zeros((n,m))
    b = np.zeros((n,m))
    c = np.zeros((n,m))
    a_norm = np.zeros((n,m))
    b_norm = np.zeros((n,m))
    c_norm = np.zeros((n,m))

    v_weighted_norm_a = np.zeros((n,m))
    v_weighted_norm_b = np.zeros((n,m))
    v_weighted_norm_c = np.zeros((n,m))

    S_weighted_a = np.zeros((n))
    S_weighted_b = np.zeros((n))
    S_weighted_c = np.zeros((n))

    v_weighted_norm_ai_a = np.zeros((m))
    v_weighted_norm_ai_b = np.zeros((m))
    v_weighted_norm_ai_c = np.zeros((m))
    v_weighted_norm_id_a = np.zeros((m))
    v_weighted_norm_id_b = np.zeros((m))
    v_weighted_norm_id_c = np.zeros((m))

    K_ai_a = np.zeros((n))
    K_ai_b = np.zeros((n))
    K_ai_c = np.zeros((n))
    K_ai_crisp = np.zeros((n))
    K_id_a = np.zeros((n))
    K_id_b = np.zeros((n))
    K_id_c = np.zeros((n))
    K_id_crisp =np.zeros ((n))

    T_a = np.zeros((n))
    T_b = np.zeros((n))
    T_c = np.zeros((n))

    f_id_a = np.zeros((n))
    f_id_b = np.zeros((n))
    f_id_c = np.zeros((n))
    f_id_crisp = np.zeros((n))
    f_ai_a = np.zeros((n))
    f_ai_b = np.zeros((n))
    f_ai_c = np.zeros((n))
    f_ai_crisp = np.zeros((n))

    f_utility = np.zeros((n))

    # Get triangular numbers
    for j in range(0, m):

        # Weights
        w_a[j] = fuzzy_weights [j,0]
        w_b[j] = fuzzy_weights [j,1]
        w_c[j] = fuzzy_weights [j,2]

        # Performance
        for i in range(0, n):
            a[i,j], b[i,j], c[i,j]  = decision_matrix_TFN[i,j]

    # Normalization
    min_columns_a = np.argmin(a, axis=0)
    max_columns_c = np.argmax(c, axis=0)

    for j in range(0, m):
        for i in range(0, n):

            a_min = a[min_columns_a[j],j]
            c_max = c[max_columns_c[j],j]
            
            if (criteria_type[j] == 'min'): # Normalization for cost criteria
                a_norm[i,j] = (a_min)/(c[i,j])
                b_norm[i,j] = (a_min)/(b[i,j])
                c_norm[i,j] = (a_min)/(a[i,j])
            
            if (criteria_type[j] == 'max'): # Normalization for benefit criteria
                a_norm[i,j] = (a[i,j])/(c_max)
                b_norm[i,j] = (b[i,j])/(c_max)
                c_norm[i,j] = (c[i,j])/(c_max)  
    
    # Weighted normalized decision matrix
    for i in range(0,n):
        for j in range(0,m):
            v_weighted_norm_a[i,j] = a_norm[i,j]*w_a[j]
            v_weighted_norm_b[i,j] = b_norm[i,j]*w_b[j]
            v_weighted_norm_c[i,j] = c_norm[i,j]*w_c[j]
        S_weighted_a[i] = np.sum(v_weighted_norm_a[i,:]) # Errata nomenclatura paper
        S_weighted_b[i] = np.sum(v_weighted_norm_b[i,:])
        S_weighted_c[i] = np.sum(v_weighted_norm_c[i,:])

    for j in range(0,m):
        
        v_weighted_norm_ai_a[j] = min(v_weighted_norm_a[:,j])
        v_weighted_norm_ai_b[j] = min(v_weighted_norm_b[:,j])
        v_weighted_norm_ai_c[j] = min(v_weighted_norm_c[:,j])

        v_weighted_norm_id_a[j] = max(v_weighted_norm_a[:,j])
        v_weighted_norm_id_b[j] = max(v_weighted_norm_b[:,j])
        v_weighted_norm_id_c[j] = max(v_weighted_norm_c[:,j])
        
    S_weighted_id_a = np.sum(v_weighted_norm_id_a[:]) # Errata nomenclatura paper
    S_weighted_id_b = np.sum(v_weighted_norm_id_b[:])
    S_weighted_id_c = np.sum(v_weighted_norm_id_c[:])

    S_weighted_ai_a = np.sum(v_weighted_norm_ai_a[:]) # Errata nomenclatura paper
    S_weighted_ai_b = np.sum(v_weighted_norm_ai_b[:])
    S_weighted_ai_c = np.sum(v_weighted_norm_ai_c[:])

    # Utility degrees K ideal and antiideal
    for i in range (0,n):
        K_ai_a[i] = S_weighted_a[i]/S_weighted_ai_c
        K_ai_b[i] = S_weighted_b[i]/S_weighted_ai_b
        K_ai_c[i] = S_weighted_c[i]/S_weighted_ai_a
        K_ai_crisp[i] = (K_ai_a[i] + 4*K_ai_b[i] +K_ai_c[i])/6

        K_id_a[i] = S_weighted_a[i]/S_weighted_id_c
        K_id_b[i] = S_weighted_b[i]/S_weighted_id_b
        K_id_c[i] = S_weighted_c[i]/S_weighted_id_a
        K_id_crisp[i] = (K_id_a[i] + 4*K_id_b[i] +K_id_c[i])/6
        
        T_a[i] = K_ai_a[i] + K_id_a[i]
        T_b[i] = K_ai_b[i] + K_id_b[i]
        T_c[i] = K_ai_c[i] + K_id_c[i]

    # Compute D    Dudas de quéquiere decir maximo
    max_columns_T_c = np.argmax(T_c)
    D_a = T_a[max_columns_T_c]
    D_b = T_b[max_columns_T_c]
    D_c = T_c[max_columns_T_c]
    D_crisp = (D_a + 4*D_b + D_c)/6

    # Utility functions corr
    for i in range(0,n):
        f_id_a [i] = K_ai_a[i]/D_crisp
        f_id_b [i] = K_ai_b[i]/D_crisp
        f_id_c [i] = K_ai_c[i]/D_crisp
        f_id_crisp[i] = (f_id_a[i] + 4*f_id_b[i] + f_id_c[i])/6

        f_ai_a [i] = K_id_a[i]/D_crisp
        f_ai_b [i] = K_id_b[i]/D_crisp
        f_ai_c [i] = K_id_c[i]/D_crisp
        f_ai_crisp[i] = (f_ai_a[i] + 4*f_ai_b[i] + f_ai_c[i])/6

        # Finally the utility function:
        f_utility[i] = (K_ai_crisp[i] + K_id_crisp[i])/(1+((1-f_id_crisp[i])/f_id_crisp[i])+((1-f_ai_crisp[i])/f_ai_crisp[i]))

    print('FMARCOS performance', f_utility)

    return

def ranking_FCOCOSO(decision_matrix_TFN,criteria_type,fuzzy_weights):
    '''
    This function computes the ranking using fuzzy COCOSO as described in:

    https://www.mdpi.com/2071-1050/14/9/4972
    '''
    # Initialization

    lambda_cocoso = 0.5 # Could be adjusted by the DM

    fuzzy_weights = np.array(fuzzy_weights)
    m = len(decision_matrix_TFN[0]) # Number of criteria
    n = len(decision_matrix_TFN) # Number of alternatives
    decision_matrix_TFN = np.array(decision_matrix_TFN)
    
    w_a = np.zeros((m))
    w_b = np.zeros((m))
    w_c = np.zeros((m))

    a = np.zeros((n,m))
    b = np.zeros((n,m))
    c = np.zeros((n,m))
    a_norm = np.zeros((n,m))
    b_norm = np.zeros((n,m))
    c_norm = np.zeros((n,m))

    S_a = np.zeros((n)) 
    S_b = np.zeros((n)) 
    S_c = np.zeros((n)) 
    
    P_a = np.zeros((n)) 
    P_b = np.zeros((n)) 
    P_c = np.zeros((n))

    f_1_a = np.zeros((n))
    f_1_b = np.zeros((n))
    f_1_c = np.zeros((n))
    f_1_crisp = np.zeros((n))

    f_2_a = np.zeros((n))
    f_2_b = np.zeros((n))
    f_2_c = np.zeros((n))
    f_2_crisp = np.zeros((n))

    f_3_a = np.zeros((n))
    f_3_b = np.zeros((n))
    f_3_c = np.zeros((n))
    f_3_crisp = np.zeros((n))

    performance_cocoso = np.zeros((n))

    # Get triangular numbers
    for j in range(0, m):

        # Weights
        w_a[j] = fuzzy_weights [j,0]
        w_b[j] = fuzzy_weights [j,1]
        w_c[j] = fuzzy_weights [j,2]

        # Performance
        for i in range(0, n):
            a[i,j], b[i,j], c[i,j]  = decision_matrix_TFN[i,j]

    # Normalization
    min_columns_a = np.argmin(a, axis=0)
    max_columns_c = np.argmax(c, axis=0)

    for j in range(0, m):
        for i in range(0, n):

            a_min = a[min_columns_a[j],j]
            c_max = c[max_columns_c[j],j]
            
            if (criteria_type[j] == 'min'): # Normalization for cost criteria
                a_norm[i,j] = (c_max-c[i,j])/(c_max-a_min)
                b_norm[i,j] = (c_max-b[i,j])/(c_max-a_min)
                c_norm[i,j] = (c_max-a[i,j])/(c_max-a_min)
            
            if (criteria_type[j] == 'max'): # Normalization for benefit criteria
                a_norm[i,j] = (a[i,j]-a_min)/(c_max-a_min)
                b_norm[i,j] = (b[i,j]-a_min)/(c_max-a_min)
                c_norm[i,j] = (c[i,j]-a_min)/(c_max-a_min)

    
    # Compute the sum of comparability arrays (S) and the sum of power 
    # weights (P) of the comparability arrays
    for i in range(0,n):
        S_a[i] = np.sum(a_norm[i,:] * w_a)
        S_b[i] = np.sum(b_norm[i,:] * w_b)
        S_c[i] = np.sum(c_norm[i,:] * w_c)
  
        P_a[i] = np.sum(a_norm[i,:] ** w_c)
        P_b[i] = np.sum(b_norm[i,:] ** w_b)
        P_c[i] = np.sum(c_norm[i,:] ** w_a)

    # Evaluation Scores
    for i in range(0,n):

        #The three fuzzy evaluation scores
        f_1_a[i] = (P_a[i] + S_a[i])/np.sum(S_c[:]+P_c[:])
        f_1_b[i] = (P_b[i] + S_b[i])/np.sum(S_b[:]+P_b[:])
        f_1_c[i] = (P_c[i] + S_c[i])/np.sum(S_a[:]+P_a[:])
        f_1_crisp[i] = (f_1_a[i] + 4*f_1_b[i] + f_1_c[i])/6 

        f_2_a[i] = S_a[i]/min(S_a)+P_a[i]/min(P_a)
        f_2_b[i] = S_b[i]/min(S_a)+P_b[i]/min(P_a)
        f_2_c[i] = S_c[i]/min(S_a)+P_c[i]/min(P_a)
        f_2_crisp[i] = (f_2_a[i] + 4*f_2_b[i] + f_2_c[i])/6

        f_3_a[i] = (lambda_cocoso*S_a[i]+(1-lambda_cocoso)*P_a[i])/(lambda_cocoso*max(S_c)+(1-lambda_cocoso)*max(P_c))
        f_3_b[i] = (lambda_cocoso*S_b[i]+(1-lambda_cocoso)*P_b[i])/(lambda_cocoso*max(S_c)+(1-lambda_cocoso)*max(P_c))
        f_3_c[i] = (lambda_cocoso*S_c[i]+(1-lambda_cocoso)*P_c[i])/(lambda_cocoso*max(S_c)+(1-lambda_cocoso)*max(P_c))
        f_3_crisp[i] = (f_3_a[i] + 4*f_3_b[i] + f_3_c[i])/6

        #Final Score
        performance_cocoso[i] = (f_1_crisp[i]*f_2_crisp[i]*f_3_crisp[i])**(1/3) + (f_1_crisp[i]+f_2_crisp[i]+f_3_crisp[i])/3

    print('FCoCoSo', performance_cocoso)

    return 

def ranking_FVIKOR(decision_matrix_TFN,criteria_type,fuzzy_weights):
    '''
    This function computes the ranking using fuzzy VIKOR as described in:

    https://www.researchgate.net/publication/268667393_Fuzzy_VIKOR_approach_Evaluating_quality_of_internet_health_information
    
    #FUZZY VIKOR APPROACH: EVALUATING QUALITY OF INTERNET
    #HEALTH INFORMATION 
    #Fuzzy VIKOR with an application to water resources planning
    #Fuzzy VIKOR as an Aid for Multiple Criteria Decision Making
    '''
    # Initialization
    fuzzy_weights = np.array(fuzzy_weights)
    m = len(decision_matrix_TFN[0]) # Number of criteria
    n = len(decision_matrix_TFN) # Number of alternatives
    decision_matrix_TFN = np.array(decision_matrix_TFN)
    
    w_a = np.zeros((m))
    w_b = np.zeros((m))
    w_c = np.zeros((m))

    a = np.zeros((n,m))
    b = np.zeros((n,m))
    c = np.zeros((n,m))

    # Normaliced fuzzy difference
    d_a = np.zeros((n,m))
    d_b = np.zeros((n,m))
    d_c = np.zeros((n,m))

    perf_a = np.zeros((m))
    perf_b = np.zeros((m))
    perf_c = np.zeros((m))

    S_a = np.zeros((n))
    S_b = np.zeros((n))
    S_c = np.zeros((n))
    S_crisp = np.zeros((n))

    R_a = np.zeros((n))
    R_b = np.zeros((n))
    R_c = np.zeros((n))
    R_crisp = np.zeros((n))

    Q_a = np.zeros((n))
    Q_b = np.zeros((n))
    Q_c = np.zeros((n))
    Q_crisp = np.zeros((n))

    # v parameter
    v = 0.5 

    # Get triangular numbers
    for j in range(0, m):

        # Weights
        w_a[j] = fuzzy_weights[j,0]
        w_b[j] = fuzzy_weights[j,1]
        w_c[j] = fuzzy_weights[j,2]

        # Performance
        for i in range(0, n):
            a[i,j], b[i,j], c[i,j]  = decision_matrix_TFN[i,j]

    # Normalization
    min_columns_a = np.argmin(a, axis=0)
    min_columns_b = np.argmin(b, axis=0)
    min_columns_c = np.argmin(c, axis=0)

    max_columns_a = np.argmax(a, axis=0)
    max_columns_b = np.argmax(b, axis=0)
    max_columns_c = np.argmax(c, axis=0)

    for j in range(0, m):
        for i in range(0, n):

            a_min = a[min_columns_a[j],j]
            b_min = b[min_columns_b[j],j]
            c_min = c[min_columns_c[j],j]

            a_max = a[max_columns_a[j],j]
            b_max = b[max_columns_b[j],j]
            c_max = c[max_columns_c[j],j]

            # Normalice fuzzy difference AFRICANO
            
            if (criteria_type[j] == 'min'): # Normalization for cost criteria
                d_a[i,j] = (a[i,j]-c_min)/(c_max-a_min)
                d_b[i,j] = (b[i,j]-b_min)/(c_max-a_min)
                d_c[i,j] = (c[i,j]-a_min)/(c_max-a_min)
            
            if (criteria_type[j] == 'max'): # Normalization for benefit criteria
                d_a[i,j] = (a_max-c[i,j])/(c_max-a_min)
                d_b[i,j] = (b_max-b[i,j])/(c_max-a_min)
                d_c[i,j] = (c_max-a[i,j])/(c_max-a_min)

    # Compute S and R
    for i in range(0,n): 

        # Compute each term of the S sumatio in order to be able to determine the fuzzy max
        for j in range(0,m):
            perf_a[j] = w_a [j] * d_a [i,j]
            perf_b[j] = w_b [j] * d_b [i,j]
            perf_c[j] = w_c [j] * d_c [i,j]

        # Fuzzy performance of the alternative            
        S_a[i] = np.sum(perf_a[:])
        S_b[i] = np.sum(perf_b[:])
        S_c[i] = np.sum(perf_c[:])
        # Fuzzy regret
        R_a[i] = max(perf_a)
        R_b[i] = max(perf_b)
        R_c[i] = max(perf_c)

    # Compute Q
    S_star_a = min(S_a)
    S_star_b = min(S_b)
    S_star_c = min(S_c)
    S_minus_c = max(S_c)

    R_star_a = min(R_a)
    R_star_b = min(R_b)
    R_star_c = min(R_c)

    R_minus_c = max(R_c)

    for i in range(0,n):
       # africano
        Q_a[i] = v*(S_a[i]-S_star_c)/(S_minus_c-S_star_a) + (1-v)*(R_a[i]-R_star_c)/(R_minus_c-R_star_a)  # Fuzzy Vikor Approach: Evaluating Quality of Internet Health Information
        Q_b[i] = v*(S_b[i]-S_star_b)/(S_minus_c-S_star_a) + (1-v)*(R_b[i]-R_star_b)/(R_minus_c-R_star_a)  
        Q_c[i] = v*(S_c[i]-S_star_a)/(S_minus_c-S_star_a) + (1-v)*(R_c[i]-R_star_a)/(R_minus_c-R_star_a)  

        # Fuzzy VIKOR as an Aid for Multiple Criteria Decision Making
        #Q_a[i] = v*(S_a[i]-S_star_a)/(S_minus_c-S_star_c) + (1-v)*(R_a[i]-R_star_a)/(R_minus_c-R_star_c)  # Note the denominator inversion
        #Q_b[i] = v*(S_b[i]-S_star_b)/(S_minus_b-S_star_b) + (1-v)*(R_b[i]-R_star_b)/(R_minus_b-R_star_b)  # Note the denominator inversion
        #Q_c[i] = v*(S_c[i]-S_star_c)/(S_minus_a-S_star_a) + (1-v)*(R_c[i]-R_star_c)/(R_minus_a-R_star_a)  # Note the denominator inversion

        S_crisp[i] = (S_a[i]+4*S_b[i]+S_c[i])/6
        R_crisp[i] = (R_a[i]+4*R_b[i]+R_c[i])/6
        Q_crisp[i] = (Q_a[i]+4*Q_b[i]+Q_c[i])/6
    
    # Order indices (decreasing)

    S_sorted_indices = np.argsort(S_crisp)
    R_sorted_indices = np.argsort(R_crisp)
    Q_sorted_indices = np.argsort(Q_crisp)

    order_S = np.argsort(S_sorted_indices) + 1
    order_R = np.argsort(R_sorted_indices) + 1
    order_Q = np.argsort(Q_sorted_indices) + 1

    print('FVIKOR Q order',order_Q)
    print('FVIKOR S order',order_S)
    print('FVIKOR R order',order_R)

    return