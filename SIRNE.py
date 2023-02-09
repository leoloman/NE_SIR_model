import poisson_pgf as p_pgf
import powerlaw_pgf as pl_pgf
import probability_generating_function as pgf
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import SimulatiViz as sv
import numpy as np
import networkx as nx
import types

class SIRNE:
    
    def __init__(self, 
                 r,
                 mu,
                 ro,
                 epsilon,
                 time, 
                 calc_g = None,
                 calc_g1 = None,
                 calc_g2 = None, 
                 probability_lambda = None,
                 G = None,
                 degree_dist = None):
        """
        Instantiate the Volz/Mercer Neighbour Exchange SIR Model
        
        args
            r: float - disease transmission to neightbor at a constant rate
            mu: float - infectious indivduals recover at a constant rate
            ro: float - rate at which neighbours are exchanged
            epsilon - fraction of infectious nodes 
            time: int - total steps 
            calc_g: types.FunctionType - function representing a probability generating function 
            calc_g1: types.FunctionType - function representing a probability generating function's first derivative 
            calc_g2: types.FunctionType - function representing a probability generating function's second derivative
            probability_lambda: - variable which may be requried as a second parameter in the probability generating function
            G: nx.Graph - a network to obtain a probability generating function from
            degree_dist: dict - a dictionary of degree distributions, where the key is the degree and the value is the probability - this will be used to construct a probability generating function
            
        
        """
        self.r = r
        self.mu = mu
        self.ro = ro
        if isinstance(G, nx.Graph):
            
            degree_dist = pgf.get_Pk(G)
            self.calc_g = pgf.get_PGF(degree_dist)
            self.calc_g1 = pgf.get_PGF_first_derivate(degree_dist)
            self.calc_g2 = pgf.get_PGF_second_derivate(degree_dist)
            
        elif isinstance(degree_dist, dict):
            self.calc_g = pgf.get_PGF(degree_dist)
            self.calc_g1 = pgf.get_PGF_first_derivate(degree_dist)
            self.calc_g2 = pgf.get_PGF_second_derivate(degree_dist)
            
        elif isinstance(calc_g, types.FunctionType) & isinstance(calc_g1, types.FunctionType) & isinstance(calc_g2, types.FunctionType) & (probability_lambda== None):
            self.calc_g = calc_g
            self.calc_g1 = calc_g1
            self.calc_g2 = calc_g2

        elif isinstance(calc_g, types.FunctionType) & isinstance(calc_g1, types.FunctionType) & isinstance(calc_g2, types.FunctionType) & (isinstance(probability_lambda, float) | isinstance(probability_lambda, int)| isinstance(probability_lambda, list)):
            self.calc_g = lambda x: calc_g(x, probability_lambda)
            self.calc_g1 = lambda x: calc_g1(x, probability_lambda)
            self.calc_g2 = lambda x: calc_g2(x, probability_lambda)
        
        # set the initial values
        self.initial_state = [1 - epsilon, # 
           epsilon / (1 - epsilon), # 
           (1 - 2*epsilon)/(1 - epsilon), #
           self.calc_g(1 - epsilon),  # Susceptible
           epsilon, #
           1 - self.calc_g(1- epsilon) # Infected
           ]
    
        self.time = list(range(time))
    
    def run_simulation(self):
        """
        Run a single simulation using scipy odeint function
        """
        self.out = sp_int.odeint(self.derv, self.initial_state, self.time, args=(self.r, self.mu, self.ro, 
                                          self.calc_g, self.calc_g1, self.calc_g2))
    

    def derv(self, x,t,rr,mm,pp, calc_g, calc_g1, calc_g2):
        #y[0]= change of theta 
        #y[1]= change of p_infec 
        #y[2]= change of p_suscep 
        #y[3]= proportion of S 
        #y[4]= change of M_I 
        #y[5]= change of I
        y=list(range(6))#zeros(6);
        y[0]=-rr*x[1]*x[0] 
        y[1]=rr*x[2]*x[1]*x[0]*calc_g2(x[0],)/calc_g1(x[0])-rr*x[1]*(1-x[1])-x[1]*mm+pp*(x[4]-x[1]) 
        y[2]=rr*x[2]*x[1]*(1-x[0]*calc_g2(x[0])/calc_g1(x[0]))+pp*(x[0]*calc_g1(x[0])/calc_g1(1)-x[2])
        y[3]=-rr*x[1]*x[0]*calc_g1(x[0]) 
        y[4]=-mm*x[4]+rr*x[1]*(x[0]**2*calc_g2(x[0])+x[0]*calc_g1(x[0])/calc_g1(1)) 
        y[5]=rr*x[1]*x[0]*calc_g1(x[0])-mm * x[5]
        return(y)
    