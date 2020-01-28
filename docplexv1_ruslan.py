# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:38:16 2019

@author: Reek's
"""

from docplex.mp.model import Model

from qiskit import BasicAer
from qiskit import *
from qiskit.aqua import run_algorithm
from qiskit.aqua.algorithms import VQE, ExactEigensolver, QAOA
from qiskit.aqua.components.optimizers import SPSA, COBYLA, L_BFGS_B
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance
from qiskit.aqua.translators.ising import docplex
from docplex.mp.model import Model
from docplex.mp.context import Context
from customized_Penalty_func import *

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
import pandas as pd
import numpy as np
#from scipy import optimize as op

#DataSet1 4 *4 Showed in presentation

vertices = [[0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]]
edges =  [[0, 3, 4, 0],
          [0, 0, 0.5, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0]]
origin = 0
destination = 3
n= len(vertices)  

######################DataSet 2 and refining undirected to directed

# DataSet 3 Note this is a bidirectional graph so no of parameters increases a bit more than expected
vertices = [[0, 4, 0, 0, 0, 0, 0, 8, 0], 
           [4, 0, 8, 0, 0, 0, 0, 11, 0], 
           [0, 8, 0, 7, 0, 4, 0, 0, 2], 
           [0, 0, 7, 0, 9, 14, 0, 0, 0], 
           [0, 0, 0, 9, 0, 10, 0, 0, 0], 
           [0, 0, 4, 14, 10, 0, 2, 0, 0], 
           [0, 0, 0, 0, 0, 2, 0, 1, 6], 
           [8, 11, 0, 0, 0, 0, 1, 0, 7], 
           [0, 0, 2, 0, 0, 0, 6, 7, 0] 
          ];

edges = [[0, 1, 0, 0, 0, 0, 0, 1, 0], 
           [1, 0, 1, 0, 0, 0, 0, 1, 0], 
           [0, 1, 0, 1, 0, 1, 0, 0, 1], 
           [0, 0, 1, 0, 1, 1, 0, 0, 0], 
           [0, 0, 0, 1, 0, 1, 0, 0, 0], 
           [0, 0, 1, 1, 1, 0, 1, 0, 0], 
           [0, 0, 0, 0, 0, 1, 0, 1, 1], 
           [1, 1, 0, 0, 0, 0, 1, 0, 1], 
           [0, 0, 1, 0, 0, 0, 1, 1, 0] 
          ];

origin = 0
destination = 5
n= len(vertices)
len(vertices)
range(0,len(edges))

# for upper mat
for i in range(0,len(edges)):
    for j in range(0,len(edges)):
        if(j < i):
            edges[i][j]=0
            vertices[i][j]=0
            
#print(edges)
#print(vertices)
vertices
##################   End of DataSet 2 refining


#N =[i for i in range(1,n)]
vert_id = [i for i in range(0,n)]
print(vert_id)
vert_m = pd.DataFrame(vertices).to_numpy()

edge_m = pd.DataFrame(edges).to_numpy()
print(vert_m)
Arcs = [(i,j) for i in vert_id for j in vert_id if i !=j]
c ={(i,j): edge_m[i,j] * vert_m[i,j] for i,j in Arcs }  
c

Arcs_modified2 =[k for k,v in c.items() if v > 0]
print(Arcs_modified2)

to_arcs =[]
from_arcs = []  
for i in Arcs_modified2:
    if (i[1] not in to_arcs ):
        to_arcs.append(i[1])
    if(i[0] not in from_arcs):
        from_arcs.append(i[0])
    

#print(Arcs_modified2)
#print(to_arcs)
        

# using doCloud Trial version Validity till 18th december
url = "https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1"
key = "api_b14a53e6-482a-44b7-b1e2-83b44d232ba0"

ctx = Context.make_default_context(url=url, key=key)

# model Objective function and constraints
mdl3 = Model(name='min_distance3',context=ctx)
x=  mdl3.binary_var_dict(Arcs_modified2,name = 'x')
#print(x)
mdl3.minimize(mdl3.sum(c[i,j] * x[i,j] for i,j in Arcs_modified2))
# origin constraint
mdl3.add_constraint(mdl3.sum(x[i,j] for i,j in Arcs_modified2 if i == origin) == 1)
#destination constraint
mdl3.add_constraint(mdl3.sum(x[i,j] for i,j in Arcs_modified2 if j == destination) == 1)

#flow balancing Eq
for i,j in Arcs_modified2:
    if j != destination:
        z = mdl3.indicator_constraint(x[i,j],mdl3.sum(x[a,b] for a,b in Arcs_modified2 if a==j) - x[i,j] == 0)
        mdl3.add_constraint_(z)
        
mdl3.solve()
print(mdl3.solve_details)
print(mdl3.solution)



qubitOp, offset = get_operator(mdl3)

ee = ExactEigensolver(qubitOp, k=1)
result = ee.run()

print('energy:', result['energy'])
print('objective:', result['energy'] + offset)

x = docplex.sample_most_likely(result['eigvecs'][0])
print('solution:', x)


#from qiskit.visualization import plot_histogram
#plot_histogram(result['eigvecs'].tolist())
seed = 1000

# with cobyla

cobyla = COBYLA(maxiter=100)
ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
vqe = VQE(qubitOp, ry,cobyla)

backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

result = vqe.run(quantum_instance)


# With Spsa and Quantum instance


spsa = SPSA(max_trials=1000)
ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
vqe = VQE(qubitOp, ry, spsa)

backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

result = vqe.run(quantum_instance)




#######VQE Result


x = docplex.sample_most_likely(result['eigvecs'][0])
print('energy:', result['energy'])
print('time:', result['eval_time'])
print('solution objective:', result['energy'] + offset)
print('solution:', x)




#from qiskit.visualization import plot_histogram
#plot_histogram(result.get)
