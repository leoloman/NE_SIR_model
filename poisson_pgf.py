import math

def calc_g(lambda_g, x):
    """
    Probability generating function for poisson distribution
    """
    
    return math.e**(lambda_g * (x-1)) 

def calc_g1(lambda_g, x):
    """
    First derivative Probability generating function for poisson distribution
    """
    
    return lambda_g * math.e**(lambda_g * (x-1)) 

def calc_g2(lambda_g, x):
    """
    Second derivative Probability generating function for poisson distribution
    """
    return (lambda_g**2) * math.e**(lambda_g * (x-1)) 