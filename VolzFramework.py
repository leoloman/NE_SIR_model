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


class VolzFramework(ABC):
    """
    args
        calc_g: types.FunctionType - function representing a probability generating function
        calc_g1: types.FunctionType - function representing a probability generating function's first derivative
        calc_g2: types.FunctionType - function representing a probability generating function's second derivative
        probability_lambda: - variable which may be requried as a second parameter in the probability generating function
        G: nx.Graph - a network to obtain a probability generating function from
        degree_dist: dict - a dictionary of degree distributions, where the key is the degree and the value is the probability - this will be used to construct a probability generating function
    """

    def __init__(
        self,
        calc_g=None,
        calc_g1=None,
        calc_g2=None,
        probability_lambda=None,
        G=None,
        degree_dist=None,
    ):
        self.epsilon = epsilon
        self.time = list(range(time))

        if isinstance(G, nx.Graph):
            degree_dist_dev_G = pgf.get_Pk(G)
            self.calc_g = pgf.get_PGF(degree_dist_dev_G)
            self.calc_g1 = pgf.get_PGF_first_derivate(degree_dist_dev_G)
            self.calc_g2 = pgf.get_PGF_second_derivate(degree_dist_dev_G)

        elif isinstance(degree_dist, dict):
            self.calc_g = pgf.get_PGF(degree_dist)
            self.calc_g1 = pgf.get_PGF_first_derivate(degree_dist)
            self.calc_g2 = pgf.get_PGF_second_derivate(degree_dist)

        elif (
            isinstance(calc_g, types.FunctionType)
            & isinstance(calc_g1, types.FunctionType)
            & (isinstance(calc_g2, types.FunctionType) | (calc_g2 == None))
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

    @abstractmethod
    def _set_initial_states(self):
        """
        Placeholder: Set the initial states
        """
        pass

    @abstractmethod
    def run_simulation(self):
        """
        Placeholder: Run a single simulation using scipy odeint function
        """
        pass

    @abstractmethod
    def ode(self):
        """
        Placeholder: function to hold all the equations in
        """
        pass
