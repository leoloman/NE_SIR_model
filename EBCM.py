import poisson_pgf as p_pgf
import powerlaw_pgf as pl_pgf
import probability_generating_function as pgf
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import SimulatiViz as sv
import numpy as np
import networkx as nx
import types
from abc import ABC, abstractmethod
from SimulatiViz import EBCMResults
from VolzFramework import VolzFramework

class EBCM(VolzFramework):
    """
    Instantiate the first Edge Based Compartmental Model
    
    Ref https://royalsocietypublishing.org/doi/10.1098/rsif.2011.0403
        https://arxiv.org/pdf/1106.6320
        https://arxiv.org/pdf/0909.4485.pdf
    """

    __doc__ += VolzFramework.__doc__


    def _set_initial_states(self):
        """
        Set the initial states
        """
        self.intial_state = [1 - self.epsilon, 0]
        
    def run_simulation(self, beta, gamma):
        """
        Run a single simulation using scipy odeint function
        """
        
        r0 = self.calc_r0(beta, gamma)
        
        output = sp_int.odeint(self.ode,
            self.initial_state,
            self.time,
            args=(beta, gamma, self.calc_g, self.calc_g1, self.calc_g2),
        )
        # add susceptible
        output = np.hstack((output, np.array([[self.calc_g(x)] for x in output[:,0]])))
        # add infected
        output = np.hstack((output, np.array([[1 - x[1] - x[2] ] for x in output])))
        
        return EBCMResults(output, dict(beta = beta, gamma = gamma), r0)
    
    def calc_r0(self, beta, gamma):
        return (beta / (beta + gamma)) * ((self.calc_g1(1)**2 - self.calc_g1(1))/self.calc_g1(1))
    
    
    def ode(self, x, beta, gamma, calc_g, calc_g1, calc_g2):
        """
        Ref https://royalsocietypublishing.org/doi/10.1098/rsif.2011.0403
            https://arxiv.org/pdf/1106.6320
            https://arxiv.org/pdf/0909.4485.pdf
        """
        
        y = list(range(2))
    
        y[0] = -beta * x[0] + beta * (calc_g1(x[0]) / calc_g1(1)) + gamma * (1-x[0])
        S = calc_g(x[0])
        I = 1 - S - x[1]
        y[1] = gamma*I
        
        return y
        
        
        
class MFSHEBCM(VolzFramework):
    
    """
    Instantiate the Mean Field Social Hetergoenity Edge Based Compartmental Model
    
    Ref https://arxiv.org/pdf/1106.6320 page 7
       
    """

    __doc__ += VolzFramework.__doc__


    def _set_initial_states(self):
        """
        Set the initial states
        """
        self.intial_state = [1 - self.epsilon, 0]
        
    def run_simulation(self, beta, gamma):
        """
        Run a single simulation using scipy odeint function
        """
        
        r0 = self.calc_r0(beta, gamma)
        
        output = sp_int.odeint(self.ode,
            self.initial_state,
            self.time,
            args=(beta, gamma, self.calc_g, self.calc_g1, self.calc_g2),
        )
        # add susceptible
        output = np.hstack((output, np.array([[self.calc_g(x)] for x in output[:,0]])))
        # add infected
        output = np.hstack((output, np.array([[1 - x[1] - x[2] ] for x in output])))
        
        return EBCMResults(output, dict(beta = beta, gamma = gamma), r0)
    
    def calc_r0(self, beta, gamma):
        return (beta / (beta + gamma)) * ((self.calc_g1(1)**2 - self.calc_g1(1))/self.calc_g1(1))
    
    
    def ode(self, x, beta, gamma, calc_g, calc_g1, calc_g2):
        """
        Ref https://arxiv.org/pdf/0909.4485.pdf page 7
        """
        
        y = list(range(2))
    
        y[0] = -beta * x[0] + beta * ((x[0]**2 * calc_g1(x[0]) )/ calc_g1(1)) - x[0]* gamma * math.log(x[0])
        S = calc_g(x[0])
        I = 1 - S - x[1]
        y[1] = gamma*I
        
        return y
    