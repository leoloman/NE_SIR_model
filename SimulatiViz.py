import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class SimResults(ABC):
    
    def __init__(self, output, param_dic, r0):
        self.output = output
        self.param_dic = param_dic
        self.r0 = r0

    @abstractmethod
    def SIR_graph(self):
        pass
    
    @abstractmethod
    def cumulative_incidence(self):
        pass
    
    @abstractmethod
    def full_simulation(self):
        pass
    
class EBCMResults(SimResults):
    
    def SIR_graph(self):
        plt.plot(self.output_3[:,1], label = 'R')
        plt.plot(self.output_3[:,2], label = 's')
        plt.plot(self.output_3[:,3], label = 'I')
        plt.legend()
    
    def cumulative_incidence(self):
        plt.plot(self.output_3[:,1]+self.output_3[:,3], label = 'R')
        plt.legend()
    
    @abstractmethod
    def full_simulation(self):
        plt.plot(self.output[:,0], label = 'theta')
        plt.plot(self.output_3[:,1], label = 'R')
        plt.plot(self.output_3[:,2], label = 's')
        plt.plot(self.output_3[:,3], label = 'I')
        plt.legend()
        
        
class NEResults(SimResults):
    
    def SIR_graph(self, N:int = None, title:str = '', 
                 figsize:tuple = (12,5), log_scale:bool = False,
                caption = ''):
    
        fig, ax = plt.subplots(figsize =figsize)
        if (N == None):
            ax.plot(self.output[:,3], label = 'S')
            ax.plot(self.output[:,5],  label = 'I')
            ax.plot(1 - (self.output[:,5]+self.output[:,3]), alpha = 0.5, label= 'R')
            ax.set_ylabel('Proportion of Nodes')

        elif (isinstance(N, int)):
            ax.plot(self.output[:,3]*N, label = 'S')
            ax.plot(self.output[:,5]*N,  label = 'I')
            ax.plot(N - ((self.output[:,5]*N)+(self.output[:,3]*N)), alpha = 0.5, label= 'R')
            ax.set_ylabel('Number of Nodes')

        if log_scale:
            ax.set_yscale('log')

        plt.figtext(0.8, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
        plt.legend(title = 'Compartment')
        plt.show()
    
    
    def cumulative_incidence(self, title:str = '', color:str = 'black', linestyle:str = ':',label = 'Cumulative Incidence', figsize:tuple = (12,5)):
        """
        Cumulative Incidence plot of a single NE simulation

        args:
            title: str - title of plot
            color: str - color of line
            linestyle: str - linestyle
            label: str - label on legend
            figsize:tuple - plot size
        """
        
        fig, ax = plt.subplots(figsize =figsize)
        ax.plot(1 - (self.output[:,5]+self.output[:,3]) + self.output[:,5], 
                alpha = 0.5, 
                c = color,
                label= label, 
                ls = linestyle)
        ax.legend()
        ax.set_title(title)
        plt.show()
        
    def full_simulation(self, title: str = '', figsize:tuple = (12,5), log_scale:bool = False):
        fig, ax = plt.subplots(figsize =figsize)
        ax.plot(self.output[:,0], alpha =0.5, label = 'Fraction degree 1 nodes sus') # change of theta
        ax.plot(self.output[:,1], label = 'Prob Sus ego to infectious alter',
                )#change of p_infec 
        ax.plot(self.output[:,2], label = 'Prob Sus ego to Sus alter')# change of p_suscep 
        ax.plot(self.output[:,3], label = 'Frac S', ls = '--') #proportion of S
        ax.plot(self.output[:,4], label = 'Infectious ego with an alter of any state')# change of M_I 
        ax.plot(self.output[:,5], alpha = 0.5, label = 'Frac I', ls = '--') # change of I
        # recovered
        ax.plot(1 - (self.output[:,5]+self.output[:,3]), alpha = 0.5, label= 'Frac R', ls = '--')
        ax.plot(1 - (self.output[:,5]+self.output[:,3]) + self.output[:,5], alpha = 0.5, c = 'black',label= 'CS Incidennce', ls = ':')
        ax.legend(title = 'Compartments', bbox_to_anchor = (1.1,0.8))
        ax.set_title(title)
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Time Step')
        if log_scale:
            ax.set_yscale('log')
        plt.show()
        
        
class SRResults(SimResults):
    
    def SIR_graph(self,  N:int = None, title:str = '', figsize:tuple = (12,5), log_scale:bool = False):
        fig, ax = plt.subplots(figsize =figsize)
        if (N == None):
            ax.plot(self.output[:,3], label = 'S')
            ax.plot(self.output[:,4],  label = 'I')
            ax.plot(1 - (self.output[:,4]+self.output[:,3]), alpha = 0.5, label= 'R')
            ax.set_ylabel('Proportion of Nodes')

        elif (isinstance(N, int)):
            ax.plot(self.output[:,3]*N, label = 'S')
            ax.plot(self.output[:,5]*N,  label = 'I')
            ax.plot(N - ((self.output[:,5]*N)+(self.output[:,3]*N)), alpha = 0.5, label= 'R')
            ax.set_ylabel('Number of Nodes')
        if log_scale:
            ax.set_yscale('log')
        plt.legend(title = 'Compartment')
        plt.show()
    
    
    def cumulative_incidence(self, title:str = '', color:str = 'black', linestyle:str = ':',label = 'Cumulative Incidence', figsize:tuple = (12,5)):
        """
        Cumulative Incidence plot of a single SR simulation

        args:
            output: np.array - result of a single SR simulation
            title: str - title of plot
            color: str - color of line
            linestyle: str - linestyle
            label: str - label on legend
            figsize:tuple - plot size
        """
        fig, ax = plt.subplots(figsize =figsize)
        ax.plot(1 - (self.output[:,4]+self.output[:,3]) + output[:,4], 
                alpha = 0.5, 
                c = color,
                label= label, 
                ls = linestyle)
        ax.legend()
        ax.set_title(title)
        plt.show()
        
    def full_simulation(self, title: str = '', figsize:tuple = (12,5), log_scale:bool = False):
        pass
    
    
    ##########################
    # Old functions
    ##########################
    
    
    
def _NE_SIR_graph(output:np.array,  N:int = None, title:str = '', 
                 figsize:tuple = (12,5), log_scale:bool = False,
                caption = ''):
    if output.shape[1] != 6:
        raise ValueError('Incorrect Simulation Output, Results are derived from the SIRNE class')
    
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
    
    if log_scale:
        ax.set_yscale('log')
        
    plt.figtext(0.8, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.legend(title = 'Compartment')
    plt.show()

    
def _SR_SIR_graph(output:np.array,  N:int = None, title:str = '', figsize:tuple = (12,5), log_scale:bool = False):
    if output.shape[1] !=5 :
        raise ValueError('Incorrect Simulation Output, Results are derived from the SIRSR class')
    fig, ax = plt.subplots(figsize =figsize)
    if (N == None):
        ax.plot(output[:,3], label = 'S')
        ax.plot(output[:,4],  label = 'I')
        ax.plot(1 - (output[:,4]+output[:,3]), alpha = 0.5, label= 'R')
        ax.set_ylabel('Proportion of Nodes')
        
    elif (isinstance(N, int)):
        ax.plot(output[:,3]*N, label = 'S')
        ax.plot(output[:,5]*N,  label = 'I')
        ax.plot(N - ((output[:,5]*N)+(output[:,3]*N)), alpha = 0.5, label= 'R')
        ax.set_ylabel('Number of Nodes')
    if log_scale:
        ax.set_yscale('log')
    plt.legend(title = 'Compartment')
    plt.show()
    
def _SR_cumulative_incidence(output:np.array,title:str = '', color:str = 'black', linestyle:str = ':',label = 'Cumulative Incidence', figsize:tuple = (12,5)):
    """
    Cumulative Incidence plot of a single SR simulation
    
    args:
        output: np.array - result of a single SR simulation
        title: str - title of plot
        color: str - color of line
        linestyle: str - linestyle
        label: str - label on legend
        figsize:tuple - plot size
    """
    if output.shape[1] !=5 :
        raise ValueError('Incorrect Simulation Output, Results are derived from the SIRSR class')
    fig, ax = plt.subplots(figsize =figsize)
    ax.plot(1 - (output[:,4]+output[:,3]) + output[:,4], 
            alpha = 0.5, 
            c = color,
            label= label, 
            ls = linestyle)
    ax.legend()
    ax.set_title(title)
    plt.show()
    
def _NE_cumulative_incidence(output:np.array,title:str = '', color:str = 'black', linestyle:str = ':',label = 'Cumulative Incidence', figsize:tuple = (12,5)):
    """
    Cumulative Incidence plot of a single NE simulation
    
    args:
        output: np.array - result of a single NE simulation
        title: str - title of plot
        color: str - color of line
        linestyle: str - linestyle
        label: str - label on legend
        figsize:tuple - plot size
    """
    if output.shape[1] != 6:
        raise ValueError('Incorrect Simulation Output, Results are derived from the SIRNE class')
        
    fig, ax = plt.subplots(figsize =figsize)
    ax.plot(1 - (output[:,5]+output[:,3]) + output[:,5], 
            alpha = 0.5, 
            c = color,
            label= label, 
            ls = linestyle)
    ax.legend()
    ax.set_title(title)
    plt.show()
    
def NE_multi_cumulative_incidence(output_list:list, title:str , label_list:list, line_style:list, color = 'black', figsize:tuple= (12,5),):
    """
    Plotting function to compare the cumulative incidence of different simulation results from the NE simulation
    
    args:
        output_list: list[np.array] a list of results from the simulations
        title: str title of the plot
        label_list: list[str] a list of strings to assign to each line in the legend
        color: str 
        line_style: list[str] a list of linestyles to assign to each line
        figsize: tuple - size of the plot
    """
    for output in output_list:
        if output.shape[1] != 6:
            raise ValueError('Incorrect Simulation Output, Results are derived from the SIRNE class')
            
    fig, ax = plt.subplots(figsize =figsize)
    for i, output in enumerate(output_list):
        ax.plot(1 - (output[:,5]+output[:,3]) + output[:,5], 
                alpha = 0.5, c = color,label= label_list[i], ls = line_style[i])
    ax.set_title(title)
    plt.show()
    
def SR_multi_cumulative_incidence(output_list:list, label_list:list, line_style:list, figsize:tuple= (12,5), color = 'black'):
    """
    Plotting function to compare the cumulative incidence of different simulation results from the SR simulation
    
    args:
        output_list: list[np.array] a list of results from the simulations
        label_list: list[str] a list of strings to assign to each line in the legend
        color: str 
        line_style: list[str] a list of linestyles to assign to each line
        figsize: tuple - size of the plot
    """
    for output in output_list:
        if output.shape[1] != 5:
            raise ValueError('Incorrect Simulation Output, Results are derived from the SIRSR class')
    fig, ax = plt.subplots(figsize =figsize)
    for i, output in enumerate(output_list):
        ax.plot(1 - (output[:,4]+output[:,3]) + output[:,4], 
                alpha = 0.5, c = color,label= label_list[i], ls = line_style[i])
    plt.show()
        

def _NE_plot_full_graph(output: np.array, title: str = '', figsize:tuple = (12,5), log_scale:bool = False):
    if output.shape[1] != 6:
        raise ValueError('Incorrect Simulation Output, Results are derived from the SIRNE class')
        
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