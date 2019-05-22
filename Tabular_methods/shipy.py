import numpy as np
import encode as cod

def shipy_policy(weights, state, h_tot, steps, epsilon, tot_turns = 400, map_size = 7):
    r = predict_reward(weights, state, h_tot, steps, tot_turns = 400, map_size = 7)
    ship_on_shipyard = state[:,:,1][state[:,:,3].astype(bool)].astype(bool)
    if (h_tot < 1000) or ship_on_shipyard:
        return False
    else:
        u = np.random.rand()
        if u < epsilon:
            C = np.random.choice([True,False])
            return C
        elif r > 0:
            return True
        else:
            return False
        
def predict_reward(weights, state, h_tot, steps, tot_turns = 400, map_size = 7):
    shipy_state = get_shipy_state(state, h_tot, steps, tot_turns = tot_turns, map_size = map_size)
    reward = poly_predict(weights, shipy_state)
    return reward

def get_shipy_state(state, h_tot, steps, tot_turns = 400, map_size = 7):
    # h_tot is the halite available
    N = np.count_nonzero(state[:,:,1]) # number of ships in the map
    t_left = tot_turns - steps # number of turns left until the end of the episode

    shipy_enc = cod.one_to_index(state[:,:,3], map_size) 
    shipy_dec = cod.decode(shipy_enc, map_size)
    s1 = shipy_dec + [1,0]
    s2 = shipy_dec + [-1,0]
    s3 = shipy_dec + [0,1]
    s4 = shipy_dec + [0,-1]
    
    s = [shipy_dec,s1,s2,s3,s4]
    mask = np.zeros((map_size,map_size)).astype(int)
    
    for x in s:
        mask[tuple(x)] = 1

    mask = mask.astype(bool)
    near_ships = state[:,:,1][mask].sum() #number of ships that in one move can go to the shipyard
    
    shipy_state = np.array([N,t_left,h_tot,near_ships])
    return shipy_state


def regress(shipy_progress):
    # shipy progress = [({s_i},R)_j]
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    l = np.array(shipy_progress)
    R = l[:,1]
    states = l[:,0]

    poly = PolynomialFeatures(2)
    d1 = len(states[0][0])
    w_len = int((d1+1)*(d1+2)/2)
    poly_States = np.zeros((len(states), w_len))
    for i in range(len(states)):
        poly_s = poly.fit_transform(states[i])
        poly_States[i] = poly_s.sum(axis=0)
    reg = LinearRegression(fit_intercept=False).fit(poly_States, R)
    w = reg.coef_
    score = reg.score(poly_States, R)
    return w, score

def poly_predict(weights, state):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    poly_state = poly.fit_transform(state.reshape(1,-1)).flatten()
    return np.dot(weights,poly_state)