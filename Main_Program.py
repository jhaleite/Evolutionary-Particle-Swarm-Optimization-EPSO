
import External_Functions
from EPSO_Bib import EPSO
from PSO_Bib import PSO

import matplotlib.pyplot as plt


obj = EPSO(8, 2, 300, -5, 5, 1.5, 1.5, 1, 0.6, 0.01)

obj.EPSO_Optimizer(External_Functions.Rosembrock_Function,'min')

print(obj.X_j)
print('+='*60)
print(obj.fx)

'''plt.figure(1)
plt.rcParams.update({'font.size': 12})
plt.plot(obj.Memory_t, obj.Memory_Gb,label=('Best Fitness EPSO'))
plt.plot(obj1.Memory_t, obj1.Memory_Gb,label=('Best Fitness PSO'))

plt.plot(obj2.Memory_t, obj2.Memory_Gb,label=('Best Fitness'))
plt.plot(obj3.Memory_t, obj3.Memory_Gb,label=('Best Fitness'))
#plt.xticks(daytime)
plt.xlabel(' Generation ')
plt.ylabel(' Best Fitness')
plt.legend()
plt.grid(True)
#plt.savefig('Power_Flow_results.png')
plt.show()'''
