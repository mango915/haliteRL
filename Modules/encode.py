import numpy as np

def one_to_index(V,L):
    """
    Parameters
    ----------
    V: LxL matrix with one entry = 1 and the others = 0
    L: linear dimension of the square matrix
    
    Assign increasing integers starting from 0 up to L**2 to an LxL matrix row by row.
    
    Returns
    -------
    integer corresponding to the non-zero element of V.
    """
    
    return np.arange(L**2).reshape((L, L))[V.astype(bool)]

def encode(v_dec, L):
    """
    Parameters
    ----------
    v_dec: list or numpy array of two integers between 0 and L
    L    : linear dimension of the square matrix
    
    Assign increasing integers starting from 0 up to L**2 to an LxL matrix row by row.
    
    Returns
    -------
    integer corresponding to the element (v_dec[0],v_dec[1]) of the encoding matrix.
    """
    V = np.arange(0,L**2).reshape((L,L))
    v_enc = V[v_dec[0],v_dec[1]] 
    return v_enc

def decode(v_enc, L):
    """
    Parameters
    ----------
    v_enc: scalar between 0 and L**2 - 1, is the encoding of a position (x,y)
    L    : linear dimension of the square matrix
    
    Assign increasing integers starting from 0 up to L**2 to an LxL matrix row by row.
    
    Returns
    -------
    numpy array containg the row and the column corresponding to the matrix element of value v_enc.
    """
    V = np.arange(0,L**2).reshape((L,L))
    v_dec = np.array([np.where(v_enc == V)[0][0],np.where(v_enc == V)[1][0]])
    return v_dec

def get_halite_vec_dec(state, q_number = 3, map_size = 7):
    """
    Parameters
    ----------
    state: [map_size,map_size,>=3] numpy array, which layers are:
            Layer 0: map halite, 
            Layer 1: ship position, 
            Layer 2: halite carried by the ships (a.k.a. cargo)
    q_number : number of quantization levels
    map_size : linear size of the squared map
    
    Returns
    -------
    quantized halite vector [𝐶,𝑂,𝑆,𝑁,𝐸,𝑊], numpy array of shape (6,)
    (where C stands for the halite carried by the ship and O for the cell occupied by the ship)
    """
    def halite_quantization(halite_vec, q_number = 3):
        """
        Creates q_number thresholds [t0,t1,t2] equispaced in the log space.
        Maps each entry of halite_vec to the corresponding level:
        if h <= t0 -> level = 0
        if t0 < h <= t1 -> level = 1
        else level = 2
        
        Parameters
        ----------
        halite_vec : numpy array which elements are numbers between 0 and 1000
        q_number : number of quantization levels

        Returns
        -------
        level : quantized halite_vec according to the q_number thresholds
        """
        # h can either be a scalar or a matrix 
        tresholds = np.logspace(1,3,q_number) # [10, 100, 1000] = [10^1, 10^2, 10^3]
        h_shape = halite_vec.shape
        h_temp = halite_vec.flatten()
        mask = (h_temp[:,np.newaxis] <= tresholds).astype(int)
        level = np.argmax(mask, axis = 1)
        return level.reshape(h_shape)

    pos_enc = one_to_index(state[:,:,1], map_size)
    pos_dec = decode(pos_enc, map_size) # decode position to access matrix by two indices
    
    ship_cargo = state[pos_dec[0],pos_dec[1],2]
    cargo_quant = halite_quantization(ship_cargo).reshape(1)[0] # quantize halite
    
    map_halite = state[:,:,0]
    halite_quant = halite_quantization(map_halite) # quantize halite
    
    halite_vector = []
    halite_vector.append(cargo_quant)
    halite_vector.append(halite_quant[pos_dec[0], pos_dec[1]])
    halite_vector.append(halite_quant[(pos_dec[0]+1)%map_size, pos_dec[1]])
    halite_vector.append(halite_quant[(pos_dec[0]-1)%map_size, pos_dec[1]])
    halite_vector.append(halite_quant[pos_dec[0], (pos_dec[1]+1)%map_size])
    halite_vector.append(halite_quant[pos_dec[0], (pos_dec[1]-1)%map_size])

    return np.array(halite_vector)

def get_halite_direction(state, map_size = 7):
    """
    Returns the direction richest in halite given the ship position.
    Works only for a single ship.
    
    Parameters
    ----------
    state: [map_size,map_size,>=3] numpy array
        Layer 0: map halite
        Layer 1: ship position 
        Layer 2: halite carried by the ships (a.k.a. cargo)
    map_size : linear size of the squared map
    
    Returns
    -------
    h_dir : int
        Dictionary to interpret the output:
        {0:'S', 1:'N', 2:'E', 3:'W'}
        
    """
    def roll_and_crop(M, shift, axis, border = 1, center = (3,3)):
        """
        Shift matrix and then crops it around the center keeping a border.

        Inputs
        ------
        M : squared matrix in numpy array
            Matrix to be rolled and cropped
        shift : int or tuple of ints
            The number of places by which elements are shifted.  If a tuple,
            then `axis` must be a tuple of the same size, and each of the
            given axes is shifted by the corresponding number.  If an int
            while `axis` is a tuple of ints, then the same value is used for
            all given axes.
        axis : int or tuple of ints, optional
            Axis or axes along which elements are shifted.  By default, the
            array is flattened before shifting, after which the original
            shape is restored.
        border : int
            Border around central cell (after the shift) to be cropped.
            The resulting area is of 2*border+1 x 2*border+1

        Parameters
        ----------
        M_cut : numpy matrix of shape (2*border+1,2*border+1)
        """
        M_temp = np.roll(M, shift = shift, axis = axis)
        M_crop = M_temp[center[0]-border:center[0]+border+1, center[1]-border:center[1]+border+1]
        return M_crop

    map_halite = state[:,:,0] # matrix with halite of each cell of the map
    shipy_pos_matrix = state[:,:,3] # matrix with 1 in presence of the shipyard, zero otherwise
    
    pos_enc = one_to_index(state[:,:,1], map_size) # ship position
    pos_dec = decode(pos_enc, map_size) # decode position to access matrix by two indices
    
    shipy_enc = one_to_index(shipy_pos_matrix, map_size) # shipyard position
    shipy_dec = decode(shipy_enc, map_size) #position_decoded 
    
    shift = (shipy_dec[0]-pos_dec[0],shipy_dec[1]-pos_dec[1])
    centered_h = np.roll(map_halite, shift = shift, axis = (0,1)) #centers map_halite on the ship
    
    mean_cardinal_h = []
    # this could be generalized to wider areas, like 5x5, but 3x3 it's enough for a 7x7 map
    perm = [(a,sh) for a in [0,1] for sh in [-2,2]] # permutations of shifts and axis to get the 4 cardinal directions
    for a,sh in perm:
        mean_h = np.mean(roll_and_crop(centered_h, shift = sh, axis = a), axis = (0,1))
        mean_cardinal_h.append(mean_h)

    mean_cardinal_h = np.array(mean_cardinal_h)
    halite_direction = np.argmax(mean_cardinal_h) #+ 1 # take the direction of the 3x3 most rich zone
    
    return halite_direction

def encode_vector(v_dec, L = 6, m = 3):
    """
    Encodes a vector of L integers ranging from 0 to m-1.
    
    Parameters
    ----------
    v_dec: list or numpy array of L integers between 0 and m
    L    : length of the vector
    
    Assign increasing integers starting from 0 up to m**L to an m-dimensional matrix "row by row".
    
    Returns
    -------
    integer corresponding to the element (v_dec[0],v_dec[1],...,v_dec[L-1]) of the encoding tensor.
    """
    T = np.arange(m**L).reshape(tuple([m for i in range(L)]))
    return T[tuple(v_dec)]

def decode_vector(v_enc, L = 6, m = 3):
    """
    Decodes an encoding for a vector of L integers ranging from 0 to m-1.
    
    Parameters
    ----------
    v_enc: scalar between 0 and m**L - 1, is the encoding of a position (x1,x2,...,xL)
    L    : length of the vector
    
    Assign increasing integers starting from 0 up to m**L to an m-dimensional matrix "row by row".
    
    Returns
    -------
    numpy array containg the indexes corresponding to the tensor element of value v_enc.
    """
    T = np.arange(m**L).reshape(tuple([m for i in range(L)]))
    return np.array([np.where(v_enc == T)[i][0] for i in range(L)])

def encode2D(v_dec, L1, L2):
    """
    Encodes a vector of 2 integers of ranges respectively L1 and L2
    e.g. the first entry must be an integer between 0 and L1-1.
     
    Parameters
    ----------
    v_dec: list or numpy array of two integers between 0 and L
    L1   : range od the first dimension
    L2   : range od the second dimension
    
    Assign increasing integers starting from 0 up to L1*L2-1 to an L1xL2 matrix row by row.
    
    Returns
    -------
    integer corresponding to the element (v_dec[0],v_dec[1]) of the encoding 2matrix.
    """
    V = np.arange(0,L1*L2).reshape((L1,L2))
    v_enc = V[tuple(v_dec)] 
    return v_enc

def decode2D(v_enc, L1, L2):
    """
    Decodes an encoding for a vector of 2 integers of ranges respectively L1 and L2. 
    
    Parameters
    ----------
    v_enc: scalar between 0 and L1*L2-1, is the encoding of a position (x,y)
    L1   : range od the first dimension
    L2   : range od the second dimension
    
    Assign increasing integers starting from 0 up to L1*L2-1 to an L1xL2 matrix row by row.
    
    Returns
    -------
    numpy array containg the row and the column corresponding to the matrix element of value v_enc.
    """
    V = np.arange(0,L1*L2).reshape((L1,L2))
    v_dec = np.array([np.where(v_enc == V)[0][0],np.where(v_enc == V)[1][0]])
    return v_dec

def encode3D(v_dec, L1, L2, L3):
    """
    Encodes a vector of 3 integers of ranges respectively L1, L2 and L3,
    e.g. the first entry must be an integer between 0 and L1-1.
     
    Parameters
    ----------
    v_dec: list or numpy array of three integers between 0 and L
    L1   : range od the first dimension
    L2   : range od the second dimension
    L3   : range od the third dimension
    
    Assign increasing integers starting from 0 up to L1*L2*L3 to an L1xL2xL3 3D-matrix "row by row".
    
    Returns
    -------
    integer corresponding to the element (v_dec[0],v_dec[1],v_dec[2]) of the encoding 3D-matrix.
    """
    V = np.arange(0,L1*L2*L3).reshape((L1,L2,L3))
    v_enc = V[tuple(v_dec)] 
    return v_enc

def decode3D(v_enc, L1, L2, L3):
    """
    Decodes an encoding for a vector of 3 integers of ranges respectively L1, L2 and L3.
    
    Parameters
    ----------
    v_enc: scalar between 0 and L1*L2*L3 - 1, is the encoding of a position (x,y)
    L1   : range od the first dimension
    L2   : range od the second dimension
    L3   : range od the third dimension
    
    Assign increasing integers starting from 0 up to L1*L2*L3 to an L1xL2xL3 3D-matrix "row by row".
    Returns
    -------
    numpy array containg the indexes corresponding to the 3D-matrix element of value v_enc.
    """
    V = np.arange(0,L1*L2*L3).reshape((L1,L2,L3))
    v_dec = np.array([np.where(v_enc == V)[0][0],np.where(v_enc == V)[1][0], np.where(v_enc == V)[2][0]])
    return v_dec

def encode_state(state, map_size = 7, h_lev = 3, n_actions = 5, debug = False):
    """
    Encode a state of the game in a unique scalar.
    
    Parameters
    ----------
     state   : [map_size,map_size,>=3] numpy array
        Layer 0: map halite
        Layer 1: ship position 
        Layer 2: halite carried by the ships (a.k.a. cargo)
    map_size : int, linear size of the squared map
    h_lev    : int, number of quantization levels of halite
    n_actions: int, number of actions that the agent can perform 
    deubg    : bool, verbose mode to debug
    
    Returns
    -------
    s_enc : int, unique encoding of the partial observation of the game state
    """
    pos_enc = one_to_index(state[:,:,1], map_size)[0] # ship position
    if debug:
        print("Ship position encoded in [0,%d]: "%(map_size**2-1), pos_enc)
    
    halvec_dec = get_halite_vec_dec(state, q_number = 3, map_size = map_size) 
    halvec_enc = encode_vector(halvec_dec) # halite vector
    if debug:
        print("Halite vector encoded in [0,%d]: "%(h_lev**6 -1), halvec_enc)
    
    haldir = get_halite_direction(state, map_size = map_size) # halite direction
    if debug:
        print("Halite direction in [0,3]: ", haldir)
    
    s_dec = np.array([pos_enc, halvec_enc, haldir])
    if debug:
        print("Decoded state: ", s_dec)
    s_enc = encode3D(s_dec, L1 = map_size**2, L2 = h_lev**6, L3 = n_actions-1)
    if debug:
        print("State encoded in [0, %d]: "%(map_size**2*h_lev**6*(n_actions-1)), s_enc, '\n')
    
    return s_enc

def scalar_to_matrix_action(action, state, map_size = 7):
    # first get the decoded position of the ship
    ship_pos_matrix = state[:,:,1]
    pos_enc = one_to_index(ship_pos_matrix, map_size)
    pos_dec = decode(pos_enc, map_size)
    # then fill a matrix of -1
    mat_action = np.full((map_size,map_size), -1)
    # finally insert the action in the pos_dec entry
    mat_action[tuple(pos_dec)] = action
    return mat_action

def sym_encode(s, map_size = 7, h_lev = 3, n_actions = 5, debug=False):
    
    # first create all the equivalent states
    
    # rotations
    s90 = np.rot90(s, k = 1)
    s180 = np.rot90(s, k = 2)
    s270 = np.rot90(s, k = 3)
    
    # reflections
    s_f = np.flip(s, axis = 1)
    s90_f = np.flip(s90, axis = 0)
    s180_f = np.flip(s180, axis = 1)
    s270_f = np.flip(s270, axis = 0)
    
    s8_dec = [s, s90, s180, s270, s_f, s90_f, s180_f, s270_f]
    
    # then encode all of them
    
    s8_enc = []
    for state in s8_dec:
        s_enc = encode_state(state, map_size = map_size, h_lev = h_lev, n_actions = n_actions, debug=False)
        s8_enc.append(s_enc)
        
    # finally returns all the encoded states 
    
    return np.array(s8_enc)

def sym_action(a):
    A = np.array([[-1,2,-1],[4,0,3],[-1,1,-1]])
    choice = np.full((3,3),a)
    M = (A==choice) # mask
    M90 = np.rot90(M, k = 1)
    M180 = np.rot90(M, k = 2)
    M270 = np.rot90(M, k = 3)

    # reflections
    M_f = np.flip(M, axis = 1)
    M90_f = np.flip(M90, axis = 0)
    M180_f = np.flip(M180, axis = 1)
    M270_f = np.flip(M270, axis = 0)

    M8 = [M, M90, M180, M270, M_f, M90_f, M180_f, M270_f]
    
    a8 = []
    for m in M8:
        a8.append(A[m][0])
    
    return a8

# multi-agent changes

def multi_scalar_to_matrix_action(actions, state, map_size = 7):
    # first get the decoded position of the ship
    ship_pos_matrix = state[:,:,1]
    ships_pos_enc = one_to_index(ship_pos_matrix, map_size)
    # then fill a matrix of -1
    mat_action = np.full((map_size,map_size), -1)
    for i in range(len(ships_pos_enc)):
        pos_dec = decode(ships_pos_enc[i], map_size)
        #print("pos_dec: ", pos_dec)
        # finally insert the action in the pos_dec entry
        mat_action[tuple(pos_dec)] = actions[i]
        
    return mat_action

def safest_dir(pos_enc, state, map_size = 7):
    # pos_enc is of a single ship
    
    ship_pos_matrix = state[:,:,1]
    shipy_enc = one_to_index(state[:,:,3], map_size) 
    shipy_dec = decode(shipy_enc, map_size)
    pos_dec = decode(pos_enc, map_size)
    shift = (shipy_dec[0]-pos_dec[0],shipy_dec[1]-pos_dec[1])
    centered = np.roll(ship_pos_matrix , shift = shift, axis = (0,1)) #centers map_halite on the ship
    
    s1 = shipy_dec + [0,1]
    s2 = shipy_dec + [0,-1]
    s3 = shipy_dec + [1,0]
    s4 = shipy_dec + [-1,0]
    s = [s1,s2,s3,s4]
    
    mask = np.zeros((map_size,map_size)).astype(int)
    for x in s:
        mask[tuple(x)] = 1

    mask = mask.astype(bool)
    near_ships = centered[mask] # N,W,E,S -> 2,4,3,1
    x = np.array([2,4,3,1])
    if near_ships.sum() < 4:
        safe_dirs = x[~near_ships.astype(bool)] # safe directions
        safest_dir = np.random.choice(safe_dirs)
    else:
        safest_dir = 0
        
    return safest_dir

def encode_multi_state(state, map_size = 7, h_lev = 3, n_actions = 5, debug = False):
    import copy
    # returns a list containing the encoded state of each ship
    ship_ids = state[:,:,4][state[:,:,1].astype(bool)]
    enc_states = []
    for i in range(len(ship_ids)):
        ID = ship_ids[i] # select one ID in order of position in the map
        mask = (state[:,:,4] == ID) # select only the position of the ship with this ID
        one_state = copy.deepcopy(state) # work with a deep copy to make changes only on that
        one_state[:,:,1][~mask] = 0 # map it to a one-ship state
        pos_enc = one_to_index(one_state[:,:,1], map_size)
        safe_dir = safest_dir(pos_enc, state, map_size = 7) # new information to encode in the multi-agent case
        # recycle the function used to encode the one-ship case by masking the other ships
        s1_enc = encode_state(one_state, map_size = map_size, h_lev = h_lev, n_actions = n_actions, debug = debug)
        n_states1 = map_size**2*h_lev**6*4 # number of possible states along s1_enc
        n_states2 = n_actions
        s_enc = encode2D(np.array([s1_enc, safe_dir]), L1 = n_states1, L2 = n_states2)
        enc_states.append(s_enc)
    return enc_states


# 4D encoding and decoding for arbitrary lengths of the four axis

def encode4D(v_dec, L1, L2, L3, L4):
    V = np.arange(0,L1*L2*L3*L4).reshape((L1,L2,L3,L4))
    v_enc = V[tuple(v_dec)] 
    return v_enc

def decode4D(v_enc, L1, L2, L3,L4):
    V = np.arange(0,L1*L2*L3*L4).reshape((L1,L2,L3,L4))
    v_dec = np.array([np.where(v_enc == V)[0][0],np.where(v_enc == V)[1][0], np.where(v_enc == V)[2][0], np.where(v_enc == V)[3][0]])
    return v_dec
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    