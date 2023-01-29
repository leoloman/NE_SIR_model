import numpy as np
import matplotlib.pyplot as plt

def SIR_graph(output:np.array, prop = True, N = None, title = '', figsize = (12,5)):
    fig, ax = plt.subplots(figsize =figsize)
    if prop:
        ax.plot(output[:,3], label = 'S')
        ax.plot(out[:,5],  label = 'I')
        ax.plot(1 - (out[:,5]+out[:,3]), alpha = 0.5, label= 'R')
        ax.set_ylabel('Proportion of Nodes')
    elif (prop == False) & (isinstance(N, int)):
        ax.plot(output[:,3]*N, label = 'S')
        ax.plot(out[:,5]*N,  label = 'I')
        ax.plot(N - ((out[:,5]*N)+(out[:,3]*N)), alpha = 0.5, label= 'R')
        ax.set_ylabel('Number of Nodes')
        
    ax.set_legend(title = 'Compartment')
    plt.show()
