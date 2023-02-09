import math

def calc_g(x, lambda_g):
    """
    Probability generating function for poisson distribution
    """
    
    return math.e**(lambda_g * (x-1)) 

def calc_g1(x, lambda_g):
    """
    First derivative Probability generating function for poisson distribution
    """
    
    return lambda_g * math.e**(lambda_g * (x-1)) 

def calc_g2(x, lambda_g):
    """
    Second derivative Probability generating function for poisson distribution
    """
    return (lambda_g**2) * math.e**(lambda_g * (x-1)) 