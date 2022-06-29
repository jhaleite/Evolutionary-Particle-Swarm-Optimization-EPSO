
import numpy as np
import EPSO_Operators


class EPSO:

    def __init__(self, Particles_Number, Variables_Number, Max_Number_Of_Generation_Allowed, Lower_Variables_Bounds, Upper_Variables_Bounds , VelocityCoeficient_CognitivePart, VelocityCoeficient_SocialPart, Inertia_value, Learning_coeficient, Noise_Parameter,  Random_StrategicParameters = 'no'):

        # Essential variables

        self.N = Particles_Number
        self.n = Variables_Number
        self.T = Max_Number_Of_Generation_Allowed
        self.Upper_Limit = Upper_Variables_Bounds
        self.Lower_Limit = Lower_Variables_Bounds
        self.tal = Learning_coeficient
        self.Constant_Noise_Parameter = Noise_Parameter
        self.Noise_Parameter = Noise_Parameter
        self.w = Inertia_value # inertia
        self.c1 = VelocityCoeficient_CognitivePart # celocity coeficient related with the cognitive part
        self.c2 = VelocityCoeficient_SocialPart # celocity coeficient related with the social part

        self.X_j = np.zeros((self.N,self.n),dtype=float) # Variables matrix of the fx function. Each line represents a particle and each collumn represents a variable
        self.V_j_parents = np.zeros((self.N,self.n),dtype=float) # Velocity matrix. Each line represents a particle and each collumn represents a variable of the velocity
        self.WX_j = np.zeros((self.N,4)) # Parameters Strategic matrix of all partiicles 
        self.WX_j_mutated = np.zeros((self.N,4)) # Mutated Parameters Strategic matrix of all partiicles 
        self.fx = np.zeros((self.N), dtype=float) # vector results of the each variables of a particle applied to a function fx

        for i in range (self.N):
            for j in range(3):
                
                if Random_StrategicParameters == 'no':
                    self.WX_j[i][0] = self.w
                    self.WX_j[i][1] = self.c1
                    self.WX_j[i][2] = self.c2
                    self.WX_j[i][3] = self.tal
                
                elif Random_StrategicParameters == 'yes':
                    self.WX_j[i][j] = np.random.uniform(0,1)
                    

        # Memory variables creation

        self.Memory_Xj = []
        self.Memory_Gb = []
        self.Memory_t = []
        self.Memory_Vj = []
        self.Memory_Pgb = []

        # Initializng random swarm

        for i in range(self.N):
            for j in range(self.n):
                if (type(self.Lower_Limit) is int) or (type(self.Lower_Limit) is float): 
                    self.X_j[i][j] = np.random.uniform(self.Lower_Limit, self.Upper_Limit)   
                else:
                    self.X_j[i][j] = np.random.uniform(self.Lower_Limit[i][j], self.Upper_Limit[i][j])
        
        self.O_j = np.copy(self.X_j)
        self.WO_j = np.copy(self.WX_j)

    def EPSO_Optimizer(self,Objective_Function, target):

        ##########################################################################################################################################################################
        # MAIN PROGRAM
        ##########################################################################################################################################################################

        t = 0 
        Pop = np.zeros((self.N*2,self.n), dtype=float)
        fPop = np.zeros((self.N*2), dtype=float)
        Plb_Pop = np.zeros((self.N*2,self.n), dtype=float)
        fPlb_Pop = np.zeros((self.N*2), dtype=float)
        Pgb_Pop = np.zeros((self.N*2,self.n), dtype=float)
        WX_Pop = np.zeros((self.N*2,self.n), dtype=float)
        O_Pop = np.zeros((self.N*2,self.n), dtype=float)
        V_Pop = np.zeros((self.N*2,self.n), dtype=float)
        fo_Pop = np.zeros((self.N*2), dtype=float)
        Plb_O_Pop = np.zeros((self.N*2,self.n), dtype=float)
        fPlb_O_Pop = np.zeros((self.N*2), dtype=float)
        Pgb_O_Pop = np.zeros((self.N*2,self.n), dtype=float)
        v_copy = np.copy(self.V_j_parents)

        while t < self.T:
            
            ###############################################################
            # STAGE 1 - PARENTS PARAMETERS CALCULATION 
            ###############################################################
            
            #  Fitness values calculation of all parents
            
            x = np.copy(self.X_j)
            Pop = np.vstack((self.X_j, x))
            WX_Pop = np.vstack((self.WX_j, self.WX_j))
            V_Pop = np.vstack((v_copy, self.V_j_parents))
            
            fPop = Objective_Function(xj=Pop) # Fitness calculation for the population

            # Updating all values of the local best of the swarm

            Plb_Pop = EPSO_Operators.LocalBest_update(Pop, fPop, Plb_Pop, fPlb_Pop, t)
            
            fPlb_Pop = Objective_Function(Plb_Pop)

            # Updating all values of the global best of the warm
            
            Pgb_Pop = EPSO_Operators.GlobalBest_Update(Pop, fPop, Pgb_Pop, target)
        
            ###############################################################
            # STAGE 2 - PARENTS STRATEGIC PARAMETERS MUTATION 
            ###############################################################
            
            WX_Pop = EPSO_Operators.EPSO_mutation(WX_Pop, self.tal, self.N) 
            
            ###############################################################
            # STAGE 3 - REPRODUCTION STAGE 
            ###############################################################
                     
            O_Pop, V_Pop = EPSO_Operators.EPSO_Reproduction(Pop, WX_Pop, V_Pop, Plb_Pop, Pgb_Pop, self.Lower_Limit, self.Upper_Limit ,self.N)

            ###############################################################
            # STAGE 4 - OFFSPRING PARAMETERS CALCULATION  
            ###############################################################
            
            # Fitness values calculation of all offsprings
             
            o = np.copy(O_Pop)
            fo_Pop = Objective_Function(xj=o) # Fitness calculation for the offsprings

            # Updating all values of the local best of the swarm

            Plb_O_Pop = EPSO_Operators.LocalBest_update(O_Pop, fo_Pop, Plb_O_Pop, fPlb_O_Pop, t)
            
            fPlb_O_Pop = Objective_Function(Plb_O_Pop) 

            # Updating all values of the global best of the warm

            Pgb_O_Pop = EPSO_Operators.GlobalBest_Update(O_Pop, fo_Pop, Pgb_O_Pop, target)
            
            ###############################################################
            # STAGE 5 - EVALUATION AND SELECTION
            ###############################################################
            
            if t == 0:
                Survivor_Variables = 0
                Survivor_FitnessValues = 0
                Survivor_StrategicParameters = 0
                Survivor_Velocity = 0
            
            Survivor_Variables, Survivor_FitnessValues, Survivor_StrategicParameters, Survivor_Velocity = EPSO_Operators.ESPO_Selection(Objective_Function, O_Pop, fo_Pop, WX_Pop, V_Pop, self.N, self.n, target)
           
            ###############################################################
            # STAGE 6 - NEW GENERATION FORMATION
            ###############################################################
                       
            self.X_j = np.copy(Survivor_Variables)
            self.WX_j = np.copy(Survivor_StrategicParameters)
            self.V_j_parents = np.copy(Survivor_Velocity)
            self.fx = np.copy(Survivor_FitnessValues)
            
            # Learning Coeficient mutation
                           
            self.Memory_Xj.append(self.X_j)
            self.Memory_Pgb.append(Pgb_O_Pop)
            self.Memory_Gb.append(min(Survivor_FitnessValues))
            self.Memory_t.append(t)
            
            t = t + 1

 