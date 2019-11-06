import numpy as np
import encode as cod


#@@@@@@@@@@@@@@@@@@@
# RL agent functions
#@@@@@@@@@@@@@@@@@@@

def greedy_policy(s_enc, q_values):
    """
    Parameters
    ----------
    s_enc : int, encoded state
    q_values : numpy matrix containing the Q-values
    
    Returns
    -------
    a : action that maximizes the Q-value given s_enc
    """
    return np.argmax(q_values[s_enc])

def e_greedy_policy(s_enc, q_values, eps = 0.01):
    """
    Parameters
    ----------
    s_enc : int, encoded state
    q_values : numpy matrix containing the Q-values
    eps : float representing the probability of choosing exploration over exploitation
    
    Returns
    -------
    a : random action with prob. eps and action that maximizes the Q-value given s_enc 
        with prob. (1-eps) 
    """
    u = np.random.rand()
    if u > eps:
        return np.argmax(q_values[s_enc])
    else:
        return np.random.randint(0, len(q_values[s_enc]))


    
def update_q_v0(s_enc, a, r, sp_enc, ap, q_values, alpha = 0.1, gamma = 1):
    """
    Update rule of Q-learning.
    
    Parameters
    ----------
    s_enc: previous encoded state
    a : action taken last turn
    r : reward obtained for last action
    sp_enc: current encoded state
    ap : action that the agent would take following the greedy policy in state 𝑠p BEFORE the update
    q_values : numpy matrix containing the Q-values
    alpha : learning rate
    gamma : discount rate
    
    Returns
    -------
    updated Q-values
    
    """
    q_values[s_enc,a] = (1-alpha)*q_values[s_enc,a] + alpha*(r + gamma*q_values[sp_enc,ap])
    return q_values

def update_q_v1(s, a, r, sp, ap, q_values, alpha = 0.1, gamma = 1, map_size = 7, h_lev = 3, n_actions = 5):
    """
    Update rule of Q-learning. Enforces Q-value = 0 for the terminal state.
    
    Parameters
    ----------
    s_enc: previous encoded state
    a : action taken last turn
    r : reward obtained for last action
    sp_enc: current encoded state
    ap : action that the agent would take following the greedy policy in state 𝑠p BEFORE the update
    q_values : numpy matrix containing the Q-values
    alpha : learning rate
    gamma : discount rate
    map_size : linear dimension of the squared map
    h_lev : number of quantization levels for the halite
    n_actions : number of actions that the agent can choose
    
    Returns
    -------
    updated Q-values
    
    """
    n_cells = map_size**2
    s_dec = cod.decode3D(s_enc, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1)
    sp_dec = cod.decode3D(sp_enc, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1)
    shipy_pos = (n_cells-1)/2 #shipyard is at the center of the map
    
    if (sp_dec[0] == shipy_pos and s_dec[0] != shipy_pos):
        # sp is terminal state -> enforce to have Q-value = 0 for all actions ap
        q_values[s_enc,a] = (1-alpha)*q_values[s_enc,a] + alpha*r 
    else:
        q_values[s_enc,a] = (1-alpha)*q_values[s_enc,a] + alpha*(r + gamma*q_values[sp_enc,ap]) # normal update
    return q_values

def update_q_v2(s, a, r, sp, ap, q_values, gamma = 1, n_cells = 49, h_lev = 3, n_actions = 5, alpha = 0.1):
    s_dec = cod.decode4D(s, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1, L4 = n_actions)
    sp_dec = cod.decode4D(sp, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1, L4 = n_actions)
    shipy_pos = (n_cells-1)/2 #shipyard is at the center of the map
    if (sp_dec[0] == shipy_pos and s_dec[0] != shipy_pos):
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*r # sp is terminal state -> enforce to have Q-value = 0 for all actions ap
        #print("Terminal value update rule executed.")
    else:
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*(r + gamma*q_values[sp,ap]) # normal update
    return q_values