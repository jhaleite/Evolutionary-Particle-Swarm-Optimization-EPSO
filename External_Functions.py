import numpy as np

def Rosembrock_Function(xj):
    
        fx = np.zeros((len(xj)), dtype=float) 
        summation = 0
        for i in range(len(xj)):   
                for j in range(len(xj[0])):
                        if j == (len(xj[0]) - 1):
                                break
                        else:
                                summation = summation + (100*(xj[i][j+1] - (xj[i][j]**2))**2 + ((1 - xj[i][j])**2)) 
                fx[i] = summation
                summation = 0
             
        return fx

def Himmelblau_Function(xj):
    
        fx = np.zeros((len(xj)), dtype=float) 
        
        for i in range(len(xj)):
                fx[i] = (((xj[i][0]**2) + xj[i][1] - 11)**2) + ((xj[i][0] + (xj[i][1]**2) - 7)**2)
                
        return fx 

def Rastrigin_Function(xj):
        
        
        fx = np.zeros(len(xj), dtype = float)
        n = len(xj[0])
        summation = 0
        for i in range(len(xj)):
                for j in range (len(xj[0])):
                        summation = summation + ((xj[i][j]**2) - 10 * np.cos(2*np.pi*xj[i][j]))
                fx[i] = 10*n + summation
                summation = 0
        return fx

def Ackley_Function(xj):
        fx = np.zeros((len(xj)), dtype=float) 
        
        for i in range(len(xj)):
                fx[i] = -20*np.exp(-0.2*np.sqrt((0.5*((xj[i][0]**2)+(xj[i][1]**2))))) - np.exp(0.5*((np.cos(2*np.pi*xj[i][0])) + np.cos(2*np.pi*xj[i][1]))) + np.exp(1) + 20                               
        return fx   