import numpy as np
import matplotlib.pyplot as plt

# some functions to represent in RGB the halite levels of the map and the ship

def color_sea(map_halite):
    #color1 = np.array([32,124,157])
    sea_colors = np.zeros((1001,3))
    for i in range(1001):
        sea_colors[i] = [32,20+i/10,50+2*i/10]
    N = len(map_halite.flatten())
    rgb_map = np.zeros((N,3))
    flatten_map = map_halite.flatten()
    for i in range(N):
        rgb_map[i] = sea_colors[int(flatten_map[i])]
    return rgb_map.reshape((map_halite.shape)+(3,))/260

def color_ship(cargo):
    #color2 = np.array([255,203,119])
    ship_colors = np.zeros((1001,3))
    for i in range(1001):
        ship_colors[i] = [(120+2*i/10)*250/320,(70+2*i/10)*200/270,(90+i/10)*120/210]
    return ship_colors[int(cargo)]/260    

def decision_map(q_s, x, y, rgb_map, map_size = 7, alpha=0.5):
    
    def Q_to_color(q_array):
        x = (q_array-q_array.min()+ 1e-5)/(q_array.max() - q_array.min() + 1e-5)
        color = np.array([[225*(1-xi),225*xi,25] for xi in x])/260
        return color
    
    pos_dict = {0:(0,0), 1:(1,0), 2:(-1,0), 3:(0,1), 4:(0,-1)}
    q_colors = Q_to_color(q_s)
    pos_dec = (x,y)
    for i in range(5):
        xi = (pos_dec[0]+pos_dict[i][0])%map_size
        yi = (pos_dec[1]+pos_dict[i][1])%map_size
        rgb_map[xi,yi] = alpha*q_colors[i]+(1-alpha)*rgb_map[xi,yi]
    return rgb_map

def upscale_colormap(rgb_mat, ps=14):
    big_rgb_mat = np.zeros((rgb_mat.shape[0]*ps, rgb_mat.shape[1]*ps,3))
    for i in range(rgb_mat.shape[0]):
        for j in range(rgb_mat.shape[1]):
            noise = np.random.normal(loc = 1, scale = 0.03, size =(ps,ps,1))
            patch = noise*rgb_mat[i,j,:]
            if patch.max() > 1:
                big_rgb_mat[i*ps:(i+1)*ps, j*ps:(j+1)*ps,:] = patch/(patch.max()+1e-4)
            else:
                big_rgb_mat[i*ps:(i+1)*ps, j*ps:(j+1)*ps,:] = patch
    return big_rgb_mat

def insert_shipy(upscale_rgb, ps=14, x=3, y=3):
    
    shipyard_object = np.array([[0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                [0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                                [0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                [0,1,1,0,0,0,1,1,0,0,0,1,1,0],
                                [0,1,1,1,0,0,1,1,0,0,1,1,1,0],
                                [0,1,1,1,1,0,1,1,0,1,1,1,1,0],
                                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                                [0,0,0,1,1,1,1,1,1,1,1,0,0,0]])

    shipy_mask = upscale_rgb[x*ps:x*ps+ps,y*ps:y*ps+ps,:]
    shipy_mask[shipyard_object.astype('bool')] = np.array([225,225,225])/255
    #shipy_mask = np.zeros((ps,ps,3))
    #shipy_mask[shipyard_object.astype('bool')] = np.array([225,225,225])/255
    #shipy_mask[~shipyard_object.astype('bool')] = np.array([32,20,50])/255
    noise = np.random.normal(loc = 1, scale = 0.05, size =(ps,ps,1))
    patch = noise*shipy_mask
    if patch.max() > 1:
        upscale_rgb[x*ps:x*ps+ps,y*ps:y*ps+ps,:] = patch/(patch.max()+1e-4)
    else:
        upscale_rgb[x*ps:x*ps+ps,y*ps:y*ps+ps,:] = patch
    
    return upscale_rgb

def insert_ship(upscale_rgb, color, x=3, y=3, ps=14):

    ship_object = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,1,1,1,0,0,0,0,0],
                            [0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                            [0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                            [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                            [0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                            [0,0,0,1,1,1,1,1,1,1,1,1,1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    ship_mask = upscale_rgb[x*ps:x*ps+ps,y*ps:y*ps+ps,:]
    ship_mask[ship_object.astype('bool')] = color
    noise = np.random.normal(loc = 1, scale = 0.05, size =(ps,ps,1))
    patch = noise*ship_mask
    if patch.max() > 1:
        upscale_rgb[x*ps:x*ps+ps,y*ps:y*ps+ps,:] = patch/(patch.max()+1e-4)
    else:
        upscale_rgb[x*ps:x*ps+ps,y*ps:y*ps+ps,:] = patch

    return upscale_rgb