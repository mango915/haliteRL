import numpy as np

# matrix to scalar encoding

def one_to_index(V,L):
    # matrix V with one entry = 1 and the others 0
    return np.arange(L**2).reshape((L, L))[V.astype(bool)]

# 2D encoding and decoding

def encode(v_dec, L):
    # v_two = [v1,v2]
    # returns the encoded version V[v1,v2] of V = np.arange(0,L)
    # L = length(all_possible_v)
    V = np.arange(0,L**2).reshape((L,L))
    v_enc = V[v_dec[0],v_dec[1]] 
    return v_enc

def decode(v_enc, L):
    V = np.arange(0,L**2).reshape((L,L))
    v_dec = np.array([np.where(v_enc == V)[0][0],np.where(v_enc == V)[1][0]])
    return v_dec

def get_halite_vec_dec(state, q_number = 3, map_size = 7):
    
    def halite_quantization(h, q_number = 3):
        # h can either be a scalar or a matrix 
        tresholds = np.logspace(1,3,q_number) # [10, 100, 1000] = [10^1, 10^2, 10^3]
        h_shape = h.shape
        h_temp = h.flatten()
        mask = (h_temp[:,np.newaxis] < tresholds).astype(int)
        level = np.argmax(mask, axis = 1)
        return level.reshape(h_shape)

    pos_enc = one_to_index(state[:,:,1], map_size)
    pos_dec = decode(pos_enc, map_size) # decode position to access matrix by two indices
    
    ship_cargo = state[pos_dec[0],pos_dec[1],2]
    #print("Halite carried: ", ship_cargo)
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
    #print("Quantized halite vector: ", halite_vector)
    return np.array(halite_vector)

# valid for encoding and decoding an array of length L whose entries can all assume only 
# the same integer values from 0 to m
def encode_tensor(v_dec, L = 6, m = 3):
    T = np.arange(m**L).reshape(tuple([m for i in range(L)]))
    return T[tuple(v_dec)]

def decode_tensor(v_enc, L = 6, m = 3):
    T = np.arange(m**L).reshape(tuple([m for i in range(L)]))
    return np.array([np.where(v_enc == T)[i][0] for i in range(L)])

def get_halite_direction(state, map_size = 7):
    
    def roll_and_cut(M, shift, axis, border = 1, center = (3,3)):
        M_temp = np.roll(M, shift = shift, axis = axis)
        M_cut = M_temp[center[0]-border:center[0]+border+1, center[1]-border:center[1]+border+1]
        return M_cut

    map_halite = state[:,:,0] # matrix with halite of each cell of the map
    
    pos_enc = one_to_index(state[:,:,1], map_size) # ship position
    pos_dec = decode(pos_enc, map_size) # decode position to access matrix by two indices
    
    shipy_pos_matrix = state[:,:,3]
    shipy_enc = one_to_index(shipy_pos_matrix, map_size) # shipyard position
    shipy_dec = decode(shipy_enc, map_size) #position_decoded 
    
    shift = (shipy_dec[0]-pos_dec[0],shipy_dec[1]-pos_dec[1])
    centered_h = np.roll(map_halite, shift = shift, axis = (0,1)) #centers map_halite on the ship
    
    mean_cardinal_h = []
    # this could be generalized to wider areas, like 5x5, but 3x3 it's enough for a 7x7 map
    perm = [(a,sh) for a in [0,1] for sh in [-2,2]] # permutations of shifts and axis to get the 4 cardinal directions
    for a,sh in perm:
        mean_h = np.mean(roll_and_cut(centered_h, shift = sh, axis = a), axis = (0,1))
        mean_cardinal_h.append(mean_h)

    mean_cardinal_h = np.array(mean_cardinal_h)
    halite_direction = np.argmax(mean_cardinal_h) #+ 1 # take the direction of the 3x3 most rich zone
    
    return halite_direction

# 3D encoding and decoding for arbitrary lengths of the three axis

def encode3D(v_dec, L1, L2, L3):
    # v_dec = [v1,v2,v3]
    # returns the encoded version V[v1,v2,v3] of V = np.arange(0,L1*L2*L3)
    #print("v_dec: ", v_dec)
    #print("L1 = %d, L2 = %d, L3 = %d"%(L1,L2,L3))
    V = np.arange(0,L1*L2*L3).reshape((L1,L2,L3))
    v_enc = V[tuple(v_dec)] 
    return v_enc

def decode3D(v_enc, L1, L2, L3):
    # v_enc = V[v1,v2,v3] 
    # V = np.arange(0,L1*L2*L3)
    # returns the decoded version v_dec = [v1,v2,v3] of V[v1,v2,v3] 
    V = np.arange(0,L1*L2*L3).reshape((L1,L2,L3))
    v_dec = np.array([np.where(v_enc == V)[0][0],np.where(v_enc == V)[1][0], np.where(v_enc == V)[2][0]])
    return v_dec

def encode_state(state, map_size = 7, h_lev = 3, n_actions = 5, debug = False):
    
    pos_enc = one_to_index(state[:,:,1], map_size)[0] # ship position
    if debug:
        print("Ship position encoded in [0,%d]: "%(map_size**2-1), pos_enc)
    
    halvec_dec = get_halite_vec_dec(state, q_number = 3, map_size = map_size) 
    halvec_enc = encode_tensor(halvec_dec) # halite vector
    if debug:
        print("Halite vector encoded in [0,%d]: "%(h_lev**6 -1), halvec_enc)
    
    haldir = get_halite_direction(state, map_size = map_size) # halite direction
    if debug:
        print("Halite direction in [1,4]: ", haldir)
    
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