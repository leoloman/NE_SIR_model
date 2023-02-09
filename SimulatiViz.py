import numpy as np
import matplotlib.pyplot as plt

def SIR_graph(output:np.array,  N:int = None, title:str = '', figsize:tuple = (12,5)):
    fig, ax = plt.subplots(figsize =figsize)
    if (N == None):
        ax.plot(output[:,3], label = 'S')
        ax.plot(output[:,5],  label = 'I')
        ax.plot(1 - (output[:,5]+output[:,3]), alpha = 0.5, label= 'R')
        ax.set_ylabel('Proportion of Nodes')
        
    elif (isinstance(N, int)):
        ax.plot(output[:,3]*N, label = 'S')
        ax.plot(output[:,5]*N,  label = 'I')
        ax.plot(N - ((output[:,5]*N)+(output[:,3]*N)), alpha = 0.5, label= 'R')
        ax.set_ylabel('Number of Nodes')
        
    plt.legend(title = 'Compartment')
    plt.show()

    
def plot_full_graph(output: np.array, title: str = '', figsize:tuple = (12,5), log_scale:bool = False):
    fig, ax = plt.subplots(figsize =figsize)
    ax.plot(output[:,0], alpha =0.5, label = 'Fraction degree 1 nodes sus') # change of theta
    ax.plot(output[:,1], label = 'Prob Sus ego to infectious alter',
            )#change of p_infec 
    ax.plot(output[:,2], label = 'Prob Sus ego to Sus alter')# change of p_suscep 
    ax.plot(output[:,3], label = 'Frac S', ls = '--') #proportion of S
    ax.plot(output[:,4], label = 'Infectious ego with an alter of any state')# change of M_I 
    ax.plot(output[:,5], alpha = 0.5, label = 'Frac I', ls = '--') # change of I
    # recovered
    ax.plot(1 - (output[:,5]+output[:,3]), alpha = 0.5, label= 'Frac R', ls = '--')
    ax.plot(1 - (output[:,5]+output[:,3]) + output[:,5], alpha = 0.5, c = 'black',label= 'CS Incidennce', ls = ':')
    ax.legend(title = 'Compartments', bbox_to_anchor = (1.1,0.8))
    ax.set_title(title)
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Time Step')
    if log_scale:
        ax.set_yscale('log')
    plt.show()