import numpy as np
import encode as cod


#@@@@@@@@@@@@@@@@@@@
# RL agent functions
#@@@@@@@@@@@@@@@@@@@

def greedy_policy(s, q_values):
    return np.argmax(q_values[s])

def e_greedy_policy(s, q_values, eps = 0.01):
    # s is encoded in input, a is encoded in output
    u = np.random.rand()
    if u > eps:
        return np.argmax(q_values[s])
    else:
        return np.random.randint(0, len(q_values[s]))

    
def update_q_v0(s, a, r, sp, ap, q_values, gamma = 1):
    q_values[s,a] = r + gamma*q_values[sp,ap]
    return q_values

def update_q_v1(s, a, r, sp, ap, q_values, gamma = 1, n_cells = 49, h_lev = 3, n_actions = 5, alpha = 0.1):
    s_dec = decode3D(s, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1)
    sp_dec = decode3D(sp, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1)
    shipy_pos = (n_cells-1)/2 #shipyard is at the center of the map
    if (sp_dec[0] == shipy_pos and s_dec[0] != shipy_pos):
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*r # sp is terminal state -> enforce to have Q-value = 0 for all actions ap
        #print("Terminal value update rule executed.")
    else:
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*(r + gamma*q_values[sp,ap]) # normal update
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