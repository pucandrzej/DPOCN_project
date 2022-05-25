import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
import pandas as pd

def c_down(c):
    """equation for c down pointing arrow
    """    
    return 1 - c

def c_up(c):
    """equation for c up pointing arrow
    """ 
    return c

def theta_down(b, c):
    """equation for theta down pointing arrow
    """ 
    return b / (2 * (1 - c))
    
def theta_up(b, c):
    """equation for theta up pointing arrow
    """ 
    return b / ((2 * c))

def PA_differential_equations(X, p, k, q_a, q_c):
    """Differential equations system for PA model. The first equation is the time evolution of proportion of positive opinion
    and the second is the time evolution of proportion of active bonds (bonds between different opinions).

    Args:
        X (list): two element list of c and b values
        p (float): probability of anticonformity, p > 0, p < 1
        k (int): size of the network
        q_a (int): size of the anticonformity group
        q_c (int): size of the conformity group

    Returns:
        array: two element array of c and b values after evolution in time step
    """    
    c = X[0]
    b = X[1]
    return np.array([ (1 - p) * (c_down(c) * theta_down(b, c) ** q_c - c_up(c) * theta_up(b ,c) ** q_c) + p * (c_down(c) * ( 1 - theta_down(b, c)) ** q_a - c_up(c) * (1 - theta_up(b, c)) ** q_a) ,
                    2 / k * (c_down(c) * ((1 - p) * theta_down(b, c) ** q_c * (k - 2 * q_c - 2 * (k - q_c) * theta_down(b, c)) + p * (1 - theta_down(b, c)) ** q_a * (k - 2 * (k - q_a) * theta_down(b, c))) + 
                    c_up(c) * ((1 - p) * theta_up(b, c) ** q_c * (k - 2 * q_c - 2 * (k - q_c) * theta_up(b, c)) + p * (1 - theta_up(b, c)) ** q_a * (k - 2 * (k - q_a) * theta_up(b, c)))) ])

def PA_algebraic_equations(vars, c, k, q_a, q_c):
    p, b = vars
    eq1 = (1 - p) * (c_down(c) * theta_down(b, c) ** q_c - c_up(c) * theta_up(b ,c) ** q_c) + p * (c_down(c) * ( 1 - theta_down(b, c)) ** q_a - c_up(c) * (1 - theta_up(b, c)) ** q_a)
    eq2 = 2 / k * (c_down(c) * ((1 - p) * theta_down(b, c) ** q_c * (k - 2 * q_c - 2 * (k - q_c) * theta_down(b, c)) + p * (1 - theta_down(b, c)) ** q_a * (k - 2 * (k - q_a) * theta_down(b, c))) + 
                        c_up(c) * ((1 - p) * theta_up(b, c) ** q_c * (k - 2 * q_c - 2 * (k - q_c) * theta_up(b, c)) + p * (1 - theta_up(b, c)) ** q_a * (k - 2 * (k - q_a) * theta_up(b, c))))
    return [eq1, eq2]

def PA_stable_fixed_points(P, t, k, q_a, q_c, IC):
    Stable_states = np.ones(np.size(P))
    for i, p in enumerate(P):
        X0 = np.array(IC)                 # initials conditions
        solution = integrate.solve_ivp(PA_differential_equations, [0, np.size(t)], X0, args = [p, k, q_a, q_c], dense_output = True)
        c, _ = solution.sol(t)
        Stable_states[i] = c[-1]
    return Stable_states

def PA_unstable_fixed_points(C, k, q_a, q_c, IC, interpolate = False, lower_interpolation_threshold = 0.01, upper_interpolation_threshold = 0.5):
    Stable_states = np.ones(10000)
    for i, c in enumerate(C):
        Stable_states[i], _ =  fsolve(PA_algebraic_equations, IC, args = (c, k, q_a, q_c))
    if interpolate:
        P_bis = np.copy(Stable_states)
        a = pd.Series(P_bis)
        a[a < lower_interpolation_threshold] = np.nan
        a[a > upper_interpolation_threshold] = np.nan
        a = a.interpolate(method='polynomial', order = 2)
        Stable_states = a.to_numpy()
    return Stable_states

def MFA_equation(c, q_a, q_c):
    X = (1 - c) * c ** q_c - c * (1 - c) ** q_c
    Y = (1 - c) ** (q_a + 1) - c ** (q_a + 1)
    return X / (X - Y)
