# TAKEN FROM: Epidemics on Networks - https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks/blob/master/EoN/analytic.py
    
# Courtesy JC MILLER

# The following functions can be used to obtain the probability generating function for a given distribution
# the get_Pk function has been included so that the distribution of a given network can be obtained
# You will have to obtain the first and second derivative of get_PGF yourself if you use this method

import numpy as np

def get_Pk(G):
    r'''
    
    TAKEN FROM: Epidemics on Networks - https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks/blob/master/EoN/analytic.py
    
    Courtesy JC MILLER
    
    Used in several places so that we can input a graph and then we 
    can call the methods that depend on the degree distribution
    :Arguments: 
    **G** networkx Graph
    
    :Returns: 
    **Pk** dict
        ``Pk[k]`` is the proportion of nodes with degree ``k``.
    '''

    Nk = Counter(dict(G.degree()).values())
    Pk = {x:Nk[x]/float(G.order()) for x in Nk.keys()}
    return Pk

def get_PGF(Pk):
    r'''
    
    TAKEN FROM: Epidemics on Networks - https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks/blob/master/EoN/analytic.py
    
    Courtesy JC MILLER
    
    Given a degree distribution (as a dict), returns the probability 
    generating function
    
    :Arguments:
    **Pk** dict
        Pk[k] is the proportion of nodes with degree k.
    :Returns: 
        
    **psi** function
            :math:`\psi(x) = \sum_k P_k[k] x^k`
    '''
    maxk = max(Pk.keys())
    ks = np.linspace(0,maxk, maxk+1)
    Pkarray = np.array([Pk.get(k,0) for k in ks])
    # print(ks)
    # print(Pkarray)
    return lambda _,  x: Pkarray.dot(x**ks)

def get_PGF_first_derivate(Pk):
    r'''
    
    TAKEN FROM: Epidemics on Networks - https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks/blob/master/EoN/analytic.py
    
    Courtesy JC MILLER
    
    Given a degree distribution (as a dict), returns the probability 
    generating function
    
    :Arguments:
    **Pk** dict
        Pk[k] is the proportion of nodes with degree k.
    :Returns: 
        
    **psi** function
            :math:`\psi(x) = \sum_k P_k[k] x^k`
    '''
    maxk = max(Pk.keys())
    # ks = np.linspace(0,maxk, maxk+1)
    ks = np.array(list(Pk.keys())[1:])
    Pkarray = np.array(list(Pk.values())[1:])
    # print(ks)
    # print(Pkarray)
    return lambda _,  x: Pkarray.dot(ks*x**(ks-1))

def get_PGF_second_derivate(Pk):
    r'''
    
    TAKEN FROM: Epidemics on Networks - https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks/blob/master/EoN/analytic.py
    
    Courtesy JC MILLER
    
    Given a degree distribution (as a dict), returns the probability 
    generating function
    
    :Arguments:
    **Pk** dict
        Pk[k] is the proportion of nodes with degree k.
    :Returns: 
        
    **psi** function
            :math:`\psi(x) = \sum_k P_k[k] x^k`
    '''
    maxk = max(Pk.keys())
    # ks = np.linspace(0,maxk, maxk+1)
    ks = np.array(list(Pk.keys())[2:])
    Pkarray = np.array(list(Pk.values())[2:])
    
    return lambda _, x: Pkarray.dot((ks*(ks-1))*x**(ks-2))

