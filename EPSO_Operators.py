import numpy as np

def EPSO_mutation(Particle_StrategicParameters, Learning_coeficient, Population_Size):
    
    N = len(Particle_StrategicParameters)
    parameters = len(Particle_StrategicParameters[0])
    Wmatrix = np.copy(Particle_StrategicParameters)
    
    #mutation_parameter = (np.random.lognormal(0,1))**Learning_coeficient
    #mutation_parameter = 1 + (Learning_coeficient*np.random.normal())
    #Wi4 = abs(Learning_coeficient * np.random.normal())
    #normal = np.random.normal()
    
    for i in range(N):
        for j in range(3):
            if i >= Population_Size:
                #Wmatrix[i][j] = Particle_StrategicParameters[i][j] * (1 + (Learning_coeficient*np.random.normal()))
                #Wmatrix[i][j] = Particle_StrategicParameters[i][j] * (np.random.lognormal(0,1))**Learning_coeficient
                Wmatrix[i][j] = Particle_StrategicParameters[i][j] + (Learning_coeficient * np.random.normal())
            
        #Wmatrix[i][3] = Particle_StrategicParameters[i][3] + (Wi4*normal)
        #Wmatrix[i][0] = Particle_StrategicParameters[i][0] * mutation_parameter
        #Wmatrix[i][1] = Particle_StrategicParameters[i][1] * mutation_parameter
        #Wmatrix[i][2] = Particle_StrategicParameters[i][2] * mutation_parameter
        #Wmatrix[i][3] = Particle_StrategicParameters[i][3] * mutation_parameter

    
    return Wmatrix

def LocalBest_update(Swarm_Variables, Swarm_FitnessValues, ParticlesLocalBestPosition, LocalBest_FitnessValues, t):
    
    N = len(Swarm_Variables)
    n = len(Swarm_Variables[0])
    X_j = np.copy(Swarm_Variables)
    Plb = np.copy(ParticlesLocalBestPosition)
    fx = np.copy(Swarm_FitnessValues)
    fx_lb =np.copy(LocalBest_FitnessValues)


    for i in range(N):
        for j in range(n):
            if t == 0:
                Plb[i][j] = X_j[i][j]     
            else:
                if fx[i] < fx_lb[i]:
                    Plb[i] = X_j[i]
  
    return Plb

def GlobalBest_Update(Swarm_Variables, Swarm_FitnessValues, ParticlesGlobalBestPosition, target):
    
    N = len(Swarm_Variables)
    X_j = np.copy(Swarm_Variables)
    Pgb = np.copy(ParticlesGlobalBestPosition)
    fx = np.copy(Swarm_FitnessValues)

    if target == 'min': # Minimization
        aux_min_fx = min(fx) # Assuming that the global best is always the lower value of all swarm
        for i in range(N):
            if fx[i] == aux_min_fx:
                aux_memory_gb = i
        for j in range(N):
            Pgb[j] = X_j[aux_memory_gb]
    
    if target == 'max': # Maximization
        aux_min_fx = max(fx) # Assuming that the global best is always the lower value of all swarm
        for i in range(N):
            if fx[i] == aux_min_fx:
                aux_memory_gb = i
        for j in range(N):
            Pgb[j] = X_j[aux_memory_gb]

    return Pgb

def EPSO_Reproduction(Particle_Variables, Particle_StrategicParameters, Particles_Velocity, LocalBest_Vector, GlobalBest_Vector, Lower_Limit, Upper_Limit, Population_Size):
    
    N = len(Particle_Variables)
    n = len(Particle_Variables[0])
    x_j = np.copy(Particle_Variables)
    WMatrix = np.copy(Particle_StrategicParameters)
    Vj = np.copy(Particles_Velocity)
    Aux_Vj = np.copy(Particles_Velocity)
    Plb = np.copy(LocalBest_Vector)
    Pgb = np.copy(GlobalBest_Vector)
   
    # Global Best perturbation
    
    aux_Pgb = np.copy(Pgb)
    for i in range(N):
        for j in range(n):
            Pgb[i][j] = aux_Pgb[i][j] + (0.001 * np.random.normal())
       
    # Velocity actualization
    
    for i in range(N):
        for j in range(n):
            if i >= Population_Size: 
                Vj[i][j] = (WMatrix[i][0]*float(Aux_Vj[i][j])) + (WMatrix[i][1]*(Plb[i][j] - x_j[i][j])) + (WMatrix[i][2]*(Pgb[i][j] - x_j[i][j]))

    # Position actualization
    
    Aux_Xj = np.copy(x_j)
    x_j = Aux_Xj + Vj
    
    # Verify if there is any variables out of the bounds
    
    for i in range(N):
        for j in range(n):
            if (type(Lower_Limit) is int) or (type(Lower_Limit) is float):
                if x_j[i][j] > Upper_Limit:
                    x_j[i][j] = Upper_Limit
                if x_j[i][j] < Lower_Limit:
                    x_j[i][j] = Lower_Limit
            else: 
                if x_j[i][j] > Upper_Limit[i][j]:
                    x_j[i][j] = Upper_Limit[i][j]
                if x_j[i][j] < Lower_Limit[i][j]:
                    x_j[i][j] = Lower_Limit[i][j]
    
    return x_j, Vj
     
def ESPO_Selection(Objective_Function, Population_Variables, Population_Fitness_Values, Population_StrategicMatrix, Population_Velocity, Population_Size, Number_of_Variables, target):
    
    N = len(Population_Variables)
    n = len(Population_Variables[0])
    Ct = np.copy(Population_Fitness_Values)
    
    
    if target == 'min':
        Ct_sorted = sorted(Ct)
    elif target == 'max':
        Ct_sorted = sorted((Ct),reverse=True)
    
    Ct_matrix = np.copy(Population_Variables)
    Ct_Wmatrix = np.copy(Population_StrategicMatrix)
    Ct_Vmatrix = np.copy(Population_Velocity)
    
    #print (Ct_matrix)

    pointer_ct = []
    
    for i in range(len(Ct)):
        for j in range(len(Ct_sorted)):
            if Ct_sorted[i] == np.copy(Ct[j]):
                pointer_ct.append(j)
                break
                
                
    #print(pointer_ct)
    #Ct = sorted (Ct)
    #print (Ct)       
    Pt_matrix = np.zeros((Population_Size, Number_of_Variables), dtype=float)
    Pt_Wmatrix = np.zeros((Population_Size, 4), dtype=float)
    Pt_VMatrix = np.zeros((Population_Size,Number_of_Variables))
    fo_ptMatrix = np.zeros((Population_Size), dtype=float)
    
    for i in range(Population_Size):
        Pt_matrix[i] = np.copy(Ct_matrix[pointer_ct[i]])
        Pt_Wmatrix[i] = np.copy(Ct_Wmatrix[pointer_ct[i]])
        Pt_VMatrix[i] = np.copy(Ct_Vmatrix[pointer_ct[i]])
    
    fo_ptMatrix = Objective_Function(Pt_matrix)
    
    
    return Pt_matrix, fo_ptMatrix, Pt_Wmatrix, Pt_VMatrix  