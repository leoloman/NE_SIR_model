import poisson_pgf as p_pgf
import powerlaw_pgf as pl_pgf
import probability_generating_function as pgf
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import SimulatiViz as sv
import numpy as np
import networkx as nx
import types

class VolzFramework:
    """
    args
        epsilon - fraction of infectious nodes
        time: int - total steps
        calc_g: types.FunctionType - function representing a probability generating function
        calc_g1: types.FunctionType - function representing a probability generating function's first derivative
        calc_g2: types.FunctionType - function representing a probability generating function's second derivative
        probability_lambda: - variable which may be requried as a second parameter in the probability generating function
        G: nx.Graph - a network to obtain a probability generating function from
        degree_dist: dict - a dictionary of degree distributions, where the key is the degree and the value is the probability - this will be used to construct a probability generating function
        """
    def __init__(
        self,
        epsilon,
        time,
        calc_g=None,
        calc_g1=None,
        calc_g2=None,
        probability_lambda=None,
        G=None,
        degree_dist=None,
    ):
        
        self.epsilon = epsilon
        
        if isinstance(G, nx.Graph):
            degree_dist = pgf.get_Pk(G)
            self.calc_g = pgf.get_PGF(degree_dist)
            self.calc_g1 = pgf.get_PGF_first_derivate(degree_dist)
            self.calc_g2 = pgf.get_PGF_second_derivate(degree_dist)

        elif isinstance(degree_dist, dict):
            self.calc_g = pgf.get_PGF(degree_dist)
            self.calc_g1 = pgf.get_PGF_first_derivate(degree_dist)
            self.calc_g2 = pgf.get_PGF_second_derivate(degree_dist)

        elif (
            isinstance(calc_g, types.FunctionType)
            & isinstance(calc_g1, types.FunctionType)
            & isinstance(calc_g2, types.FunctionType)
            & (probability_lambda == None)
        ):
            self.calc_g = calc_g
            self.calc_g1 = calc_g1
            self.calc_g2 = calc_g2

        elif (
            isinstance(calc_g, types.FunctionType)
            & isinstance(calc_g1, types.FunctionType)
            & isinstance(calc_g2, types.FunctionType)
            & (
                isinstance(probability_lambda, float)
                | isinstance(probability_lambda, int)
                | isinstance(probability_lambda, list)
            )
        ):
            self.calc_g = lambda x: calc_g(x, probability_lambda)
            self.calc_g1 = lambda x: calc_g1(x, probability_lambda)
            self.calc_g2 = lambda x: calc_g2(x, probability_lambda)
            
        self._set_intial_states()

    def _set_inital_states(self):
        """
        Placeholder: Set the initial states
        """
    
    def run_simulation(self):
        """
        Placeholder: Run a single simulation using scipy odeint function
        """
        pass
    
    def ode(self):
        """
        Placeholder: function to hold all the equations in
        """
        

class SIRNE(VolzFramework):
    """
    Instantiate the  Volz/Mercer Neighbour Exchange SIR Model (NE)
    """
       
    __doc__ += VolzFramework.__doc__
        
    def _set_inital_states(self):
        """
        
        """
         # set the initial values
        self.initial_state = [
            1 - self.epsilon,  #
            self.epsilon / (1 - self.epsilon),  #
            (1 - 2 * self.epsilon) / (1 - self.epsilon),  #
            self.calc_g(1 - self.epsilon),  # Susceptible
            self.epsilon,  #
            1 - self.calc_g(1 - self.epsilon),  # Infected
        ]

    def run_simulation(self, r: float, mu: float, rho: float) -> np.array:
        """
        Run a single simulation using scipy odeint function

        args:
            r: float - disease transmission to neightbor at a constant rate
            mu: float - infectious indivduals recover at a constant rate
            rho: float - rate at which neighbours are exchanged
            
        return:
            np.array - time x 5 matrix representing each state
        """

        return sp_int.odeint(
            self.ode,
            self.initial_state,
            self.time,
            args=(r, mu, rho, self.calc_g, self.calc_g1, self.calc_g2),
        )

    
    def ode(self, x, t, rr, mm, pp, calc_g, calc_g1, calc_g2):
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
    """
    __doc__ += VolzFramework.__doc__
    
    def _set_inital_states(self):
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
    
    def run_simulation(self, r: float, mu: float) -> np.array:
        """
        Run a single simulation using scipy odeint function

        args:
            r: float - disease transmission to neightbor at a constant rate
            mu: float - infectious indivduals recover at a constant rate
            
        return:
            np.array - time x 5 matrix representing each state
        """

        return sp_int.odeint(
            self.ode,
            self.initial_state,
            self.time,
            args=(r, mu, self.calc_g, self.calc_g1, self.calc_g2),
        )
    
    
    
    def ode(self, x, t, rr, mm, calc_g, calc_g1, calc_g2):
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