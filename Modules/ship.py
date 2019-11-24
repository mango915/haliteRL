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
    s_dec = cod.decode3D(s, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1)
    sp_dec = cod.decode3D(sp, L1 = n_cells, L2 = h_lev**6, L3 = n_actions-1)
    shipy_pos = (n_cells-1)/2 #shipyard is at the center of the map
    
    if (sp_dec[0] == shipy_pos and s_dec[0] != shipy_pos):
        # sp is terminal state -> enforce to have Q-value = 0 for all actions ap
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*r 
    else:
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*(r + gamma*q_values[sp,ap]) # normal update
    return q_values

def update_q_v3(s, a, r, sp, ap, q_values, alpha = 0.1, gamma = 1, map_size = 7, h_lev = 3, cargo_lev = 4, n_actions = 5):
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
    s_dec = cod.decode3D(s, L1 = n_cells, L2 = cargo_lev*h_lev**5, L3 = n_actions-1)
    sp_dec = cod.decode3D(sp, L1 = n_cells, L2 = cargo_lev*h_lev**5, L3 = n_actions-1)
    shipy_pos = (n_cells-1)/2 #shipyard is at the center of the map
    
    if (sp_dec[0] == shipy_pos and s_dec[0] != shipy_pos):
        # sp is terminal state -> enforce to have Q-value = 0 for all actions ap
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*r 
    else:
        q_values[s,a] = (1-alpha)*q_values[s,a] + alpha*(r + gamma*q_values[sp,ap]) # normal update
    return q_values

def play_episode(q_values, eps, NUM_PLAYERS, MAP_SIZE, TOT_TURNS, N_ACTIONS, H_LEV,
                 STD_REWARD,LEARNING_RATE, DISCOUNT_FACTOR, verbose = False):
    """
    Trains the agent by playing one episode of halite.
    
    Parameters
    ----------
    q_values         : numpy array 
        Contains the Q-values
    eps              : float 
        Represents a probability, must be in [0,1], controls the probability of exploring instead of exploting
    NUM_PLAYERS      : int
    MAP_SIZE         : int
    TOT_TURNS        : int
    N_ACTIONS        : int
    H_LEV            : int
    STD_REWARD       : float
        Baseline reward given to the agent when does not deposit halite to the shipyard
    LEARNING_RATE    : float
    DISCOUNT_FACTOR  : float
        Must be greater than 0 but smaller than 1. Suggested 1-1/TOT_TURNS or 1
    verbose          : bool
        Prints halite of the player at each turn of the game
        
    Returns
    -------
    q_values         : numpy array 
        Updated Q-values
    reward           : float
        Reward obtained in this episode. 
    collected_halite : float
        Halite collected by the agent.
    passages         : int
        Number of passages of the ship through the shipyard.
    """
    
    import sys
    sys.path.insert(0, "../Environment/")
    import halite_env as Env
    env = Env.HaliteEnv(NUM_PLAYERS, MAP_SIZE, episode_lenght = TOT_TURNS) # init environment
    steps = 0
    reward = 0 # cumulative reward of the episode
    passages = 0 # number of times the ship passes through the shipyard
    
    # first mandatory step
    steps = steps + 1
    if verbose:
        print("\nStep number %d:"%steps)
    action_matrix = np.full((MAP_SIZE,MAP_SIZE), -1) # no ship, no action
    shipyard_action = True # initially always choose to create a ship
    # returns the matricial state, the array of players halite and a flag that is true if it's the final turn
    state, players_halite, finish, _ = env.step(action_matrix, makeship = shipyard_action) 
    #print("Cargo layer: \n", state[:,:,2])
    current_halite = players_halite[0][0]
    s_enc = cod.encode_state(state, map_size = MAP_SIZE, h_lev = H_LEV, n_actions = N_ACTIONS, debug=False)

    while True:
        steps = steps + 1
        if verbose:
            print("\nStep number %d:"%steps)
            print("Current halite: ", current_halite)
        a_enc = e_greedy_policy(s_enc, q_values, eps = eps)
        a_mat = cod.scalar_to_matrix_action(a_enc, state, map_size = MAP_SIZE) #convert the action in matricial form

        # submit the action and get the new state
        state, players_halite, finish, _ = env.step(a_mat, makeship = False) 

        new_halite = players_halite[0][0]

        # compute the 1-ship reward as the halite increment of the player divided by the max halite 
        # plus a standard negative reward 
        r = (new_halite - current_halite)/1000 + STD_REWARD

        sp_enc = cod.encode_state(state, map_size = MAP_SIZE, h_lev = H_LEV, n_actions = N_ACTIONS, debug=False)
        reward += r # cumulative reward of the episode

        # adds 1 to passages if the current position of the ship coincides with that of the shipyard
        # whereas the previous position didn't
        s_dec = cod.decode3D(s_enc, L1 = MAP_SIZE**2, L2 = H_LEV**6, L3 = N_ACTIONS-1)
        sp_dec = cod.decode3D(sp_enc, L1 = MAP_SIZE**2, L2 = H_LEV**6, L3 = N_ACTIONS-1)
        shipy_pos = (MAP_SIZE**2-1)/2 #shipyard is at the center of the map
        if (sp_dec[0] == shipy_pos and s_dec[0] != shipy_pos):
            passages = passages +1
                
        a_temp_enc = greedy_policy(sp_enc, q_values) # simulate the best action in the new state (before update)

        # update Q-values
        q_values = update_q_v1(s_enc, a_enc, r, sp_enc, a_temp_enc, q_values, alpha = LEARNING_RATE,
                    gamma = DISCOUNT_FACTOR, map_size = MAP_SIZE, h_lev = H_LEV, n_actions = N_ACTIONS)

        # update states and halite
        s_enc = sp_enc
        current_halite = new_halite

        if (finish == True) or (steps >= 400):
            if verbose:
                print("\nEnd episode.")
            break
    collected_halite = current_halite - 4000
    return q_values, reward, collected_halite, passages

def play_episode_v1(q_values, eps, NUM_PLAYERS, MAP_SIZE, TOT_TURNS, N_ACTIONS, MAP_H_THRESHOLDS, CARGO_THRESHOLDS,
                 R0, R1, LEARNING_RATE, DISCOUNT_FACTOR, verbose = False, debug = False):
    """
    Trains the agent by playing one episode of halite.
    
    Parameters
    ----------
    q_values         : numpy array 
        Contains the Q-values
    eps              : float 
        Represents a probability, must be in [0,1], controls the probability of exploring instead of exploting
    NUM_PLAYERS      : int
    MAP_SIZE         : int
    TOT_TURNS        : int
    N_ACTIONS        : int
    H_LEV            : int
    STD_REWARD       : float
        Baseline reward given to the agent when does not deposit halite to the shipyard
    LEARNING_RATE    : float
    DISCOUNT_FACTOR  : float
        Must be greater than 0 but smaller than 1. Suggested 1-1/TOT_TURNS or 1
    verbose          : bool
        Prints halite of the player at each turn of the game
        
    Returns
    -------
    q_values         : numpy array 
        Updated Q-values
    reward           : float
        Reward obtained in this episode. 
    collected_halite : float
        Halite collected by the agent.
    passages         : int
        Number of passages of the ship through the shipyard.
    """
    import sys
    sys.path.insert(0, "../Environment/")
    import halite_env as Env
    # define some derived constants
    H_LEV = len(MAP_H_THRESHOLDS)
    CARGO_LEV = len(CARGO_THRESHOLDS)
    
    verbose_print = print if verbose else lambda *args, **kwargs : None # define verbose printing function
    verbose_print("Verbose: ", verbose)
    
    env = Env.HaliteEnv(NUM_PLAYERS, MAP_SIZE, episode_lenght = TOT_TURNS) # init environment
    steps = 0
    reward = 0 # cumulative reward of the episode
    passages = 0 # number of times the ship passes through the shipyard
    
    # first mandatory step
    steps = steps + 1
    verbose_print("\nStep number %d:"%steps)
    action_matrix = np.full((MAP_SIZE,MAP_SIZE), -1) # no ship, no action
    shipyard_action = True # initially always choose to create a ship
    # returns the matricial state, the array of players halite and a flag that is true if it's the final turn
    state, players_halite, finish, _ = env.step(action_matrix, makeship = shipyard_action) 
    current_halite = players_halite[0][0]
    s_enc = cod.encode_state_v1(state, MAP_H_THRESHOLDS, CARGO_THRESHOLDS, map_size = MAP_SIZE, debug = debug)

    while True:
        steps = steps + 1
        verbose_print("\nStep number %d:"%steps)
        verbose_print("Current halite: ", current_halite)
        a_enc = e_greedy_policy(s_enc, q_values, eps = eps)
        a_mat = cod.scalar_to_matrix_action(a_enc, state, map_size = MAP_SIZE) #convert the action in matricial form

        # submit the action and get the new state
        state, players_halite, finish, _ = env.step(a_mat, makeship = False) 

        new_halite = players_halite[0][0]

        # compute the 1-ship reward as the halite increment of the player divided by the max halite 
        # plus a standard negative reward 
        #---------------------------------------------------------------------------------------------------------
        if new_halite == current_halite:
            r =  R0
        else:
            r = (new_halite - current_halite)/1000 + R1 # this is the change in the code
        #---------------------------------------------------------------------------------------------------------
        sp_enc = cod.encode_state_v1(state, MAP_H_THRESHOLDS, CARGO_THRESHOLDS, map_size = MAP_SIZE, debug = debug)
        reward += r # cumulative reward of the episode

        # adds 1 to passages if the current position of the ship coincides with that of the shipyard
        # whereas the previous position didn't
        s_dec = cod.decode3D(s_enc, L1 = MAP_SIZE**2, L2 = CARGO_LEV*H_LEV**5, L3 = N_ACTIONS-1)
        sp_dec = cod.decode3D(sp_enc, L1 = MAP_SIZE**2, L2 = CARGO_LEV*H_LEV**5, L3 = N_ACTIONS-1)
        shipy_pos = (MAP_SIZE**2-1)/2 #shipyard is at the center of the map
        if (sp_dec[0] == shipy_pos and s_dec[0] != shipy_pos):
            passages = passages +1
            verbose_print("Passage number : ", passages)
                
        a_temp_enc = greedy_policy(sp_enc, q_values) # simulate the best action in the new state (before update)

        # update Q-values
        q_values = update_q_v3(s_enc, a_enc, r, sp_enc, a_temp_enc, q_values, 
                                    alpha = LEARNING_RATE, gamma = DISCOUNT_FACTOR, map_size = MAP_SIZE, 
                                    h_lev = H_LEV, cargo_lev = CARGO_LEV, n_actions = N_ACTIONS)

        # update states and halite
        s_enc = sp_enc
        current_halite = new_halite

        if (finish == True) or (steps >= 400):
            verbose_print("\nEnd episode.")
            break
    collected_halite = current_halite - 4000
    return q_values, reward, collected_halite, passages

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
