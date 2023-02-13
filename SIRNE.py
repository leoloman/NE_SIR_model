import poisson_pgf as p_pgf
import powerlaw_pgf as pl_pgf
import probability_generating_function as pgf
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import SimulatiViz as sv
import numpy as np
import networkx as nx
import types
from VolzFramework import VolzFramework
from SimulatiViz import SRResults, NEResults

class SIRNE(VolzFramework):
    """
    Instantiate the  Volz/Mercer Neighbour Exchange SIR Model (NE)
    
    Ref: https://royalsocietypublishing.org/doi/10.1098/rspb.2007.1159
    """

    __doc__ += VolzFramework.__doc__

    def _set_initial_states(self, epsilon):
        """ """
        # set the initial values
        return [
            1 - epsilon,  #
            epsilon / (1 - epsilon),  #
            (1 - 2 * epsilon) / (1 - epsilon),  #
            self.calc_g(1 - epsilon),  # Susceptible
            epsilon,  #
            1 - self.calc_g(1 - epsilon),  # Infected
        ]

    def run_simulation(self, r: float, mu: float, rho: float, epsilon: float, timesteps: int) -> np.array:
        """
        Run a single simulation using scipy odeint function

        args:
            r: float - disease transmission to neightbor at a constant rate
            mu: float - infectious indivduals recover at a constant rate
            rho: float - rate at which neighbours are exchanged
            epsilon - fraction of infectious nodes
            timesteps: int - total steps

        return:
            np.array - time x 5 matrix representing each state
        """
        r0 = self.calc_r0(r, mu, rho)
        
        time = list(range(timestemps))
                    
        inital_state = self._set_initial_states(epsilon)
        
        output = sp_int.odeint(
            self.ode,
            initial_state,
            time,
            args=(r, mu, rho, self.calc_g, self.calc_g1, self.calc_g2),
        )
        
        return NEResults(output, dict(r = r, mu = mu, rho = rho, epsilon = epsilon), r0)
    
    def calc_r0(self, r:float, mu:float, rho:float):
        return (r/mu) * ((self.calc_g2(1)/self.calc_g1(1)) * (mu + rho) + rho)

    def ode(self, x, t, rr, mm, pp, calc_g, calc_g1, calc_g2):
        """
        Ref: https://royalsocietypublishing.org/doi/10.1098/rspb.2007.1159
        """
        # y[0]= change of theta
        # y[1]= change of p_infec
        # y[2]= change of p_suscep
        # y[3]= proportion of S
        # y[4]= change of M_I
        # y[5]= change of I
        y = list(range(6))  # zeros(6);
        y[0] = -rr * x[1] * x[0]
        y[1] = (
            rr * x[2] * x[1] * x[0] * calc_g2(x[0]) / calc_g1(x[0])
            - rr * x[1] * (1 - x[1])
            - x[1] * mm
            + pp * (x[4] - x[1])
        )
        y[2] = rr * x[2] * x[1] * (1 - x[0] * calc_g2(x[0]) / calc_g1(x[0])) + pp * (
            x[0] * calc_g1(x[0]) / calc_g1(1) - x[2]
        )
        y[3] = -rr * x[1] * x[0] * calc_g1(x[0])
        y[4] = -mm * x[4] + rr * x[1] * (
            x[0] ** 2 * calc_g2(x[0]) + x[0] * calc_g1(x[0]) / calc_g1(1)
        )
        y[5] = rr * x[1] * x[0] * calc_g1(x[0]) - mm * x[5]
        return y
    
    

class SIRSR(VolzFramework):
    """
    Instantiate the Volz Semi Random Static Network (SR) Model
    
    Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7080148/pdf/285_2007_Article_116.pdf
    """

    __doc__ += VolzFramework.__doc__

    def _set_initial_states(self, epsilon):
        """ 
        """
        # set the initial values
        self.initial_state = [
            1 - epsilon,  # theta
            epsilon / (1 - epsilon),  # force of infection
            (1 - 2 * epsilon)
            / (
                1 - epsilon
            ),  # represents the probability that an arc with a susceptible ego has a susceptible alter
            self.calc_g(1 - epsilon),  #
            1 - self.calc_g(1 - epsilon),  #
        ]

    def run_simulation(self, r: float, mu: float, epsilon: float, timesteps: int) -> np.array:
        """
        Run a single simulation using scipy odeint function

        args:
            r: float - disease transmission to neightbor at a constant rate
            mu: float - infectious indivduals recover at a constant rate
            epsilon - fraction of infectious nodes
            timesteps: int - total steps

        return:
            np.array - time x 5 matrix representing each state
        """
        
        time = list(range(timesteps)
                    
        inital_state = self._set_initial_states(epsilon)
            
        output = sp_int.odeint(
            self.ode,
            initial_state,
            time,
            args=(r, mu, self.calc_g, self.calc_g1, self.calc_g2),
        )
        
        return SRResults(output, dict(r = r, mu = mu, epsilon = epsilon),  None )

    def ode(self, x, t, rr, mm, calc_g, calc_g1, calc_g2):
        """
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7080148/pdf/285_2007_Article_116.pdf
        """
        # x[0] - theta
        # x[1] pi
        # x[2] ps
        # x[3] s
        # x[4] I

        y = list(range(5))  # zeros(6);

        y[0] = -rr * x[1] * x[0]

        y[1] = (
            rr * x[2] * x[1] * x[0] * (calc_g2(x[0]) / calc_g1(x[0]))
            - rr * x[1] * (1 - x[1])
            - x[1] * mm
        )

        y[2] = rr * x[2] * x[1] * (1 - x[0] * (calc_g2(x[0]) / calc_g1(x[0])))

        y[3] = -rr * x[1] * x[0] * calc_g1(x[0])
        y[4] = rr * x[1] * x[0] * calc_g1(x[0]) - mm * x[4]
        return y
