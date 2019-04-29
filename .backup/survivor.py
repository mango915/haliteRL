import numpy as np
import halite_env as he

debug = True
map_size = 6
n_cells = map_size ** 2
nlayers = 7
num_ships = 10

map_generator = he.MapGenerator()
mapp = map_generator.dummy_map(6, 10)

action = he.dummy_action(mapp, 10)
rolledSA = he.roll_state(mapp, action)


directions = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
ROLL = rolledSA.copy()
ROLL = ROLL.reshape(5, n_cells, nlayers)
ROLL = np.swapaxes(ROLL, 0, 1)

STATE = ROLL[:, :, :-1]
ACTION = ROLL[:, :, -1:]

# check final number of ships for every cell
S = (directions[np.newaxis,] == ACTION).sum(axis=1)


# ACTION FIVE
# check not a shipyard
mask_not_shipy = STATE[:, 0, 2] != 1
# check create dropoff
mask_action_five = ACTION[:, 0, 0] == 5

# TODO: STATE[:,0,0] add to player's halite
# check two previous checks together
mask_five_not_shipy = np.all((mask_not_shipy, mask_action_five), axis=0)
# remove cell's halite
STATE[mask_five_not_shipy, 0, 0] = 0
# create dropoff
STATE[mask_five_not_shipy, 0, 2] = -1

# COLLISION (S==2)
# check cells with collision
mask_collision = (S > 1)[:, 0]  # (S>1)[:,0] <- [:,0] is for broadcasting
# check incoming ships
mask_arrivals = (ACTION == directions[np.newaxis, ...])[:, :, 0]
# halite from neigbours where there is a collision but does not check yet if it is coming
potential_drop = STATE[:, :, 1][mask_collision].copy()
# check coming
mask_drop = mask_arrivals[mask_collision]
# remove cargo halite that is not coming(this is just a copy)
potential_drop[~mask_drop] = 0
# add dropped halite to ecean
# TODO add dropoff case
STATE[mask_collision, 0, 0] += potential_drop.sum(axis=1)
# remove cargo halite
STATE[mask_collision, 0, 1] = 0
# remove ships
STATE[mask_collision, 0, 3] = 0

# ACTION (S==1)
# check non interacting moves
mask_action = (S == 1)[:, 0]  # (S>1)[:,0] <- [:,0] is for broadcasting
# stay still move
mask_stay = np.all((ACTION[:, 0, 0] == 0, mask_action), axis=0)
# calculate 25% of cell's halite
potential_gain = np.round(STATE[:, 0, 0] * 0.25)
# check actual cargos
potential_cargos = STATE[:, 0, 1]
# check fullness
mask_not_full = (potential_cargos + potential_gain) <= 1000
# unify stay and not full
mask_stay_not_full = np.all((mask_stay, mask_not_full), axis=0)
# unify stay and full
mask_stay_full = np.all((mask_stay, ~mask_not_full), axis=0)
# take all 25% of the halite
STATE[mask_stay_not_full, 0, 0] -= potential_gain[mask_stay_not_full]
STATE[mask_stay_not_full, 0, 1] += potential_gain[mask_stay_not_full]
# take halite only for fill the space left
space_left = 1000 - STATE[mask_stay_full, 0, 1]
STATE[mask_stay_full, 0, 0] -= space_left
STATE[mask_stay_full, 0, 1] = 1000

# movement step
mask_coming_ships = np.squeeze((directions[np.newaxis,] == ACTION), axis=2)[mask_action]
# ship arrive
STATE[mask_action, 0, 3] = 1
# cargo arrive
STATE[mask_action, 0, 1] = STATE[:, :, 1][mask_action][mask_coming_ships]

# VOID (S==0)
# check no ships in cell
mask_void = (S == 0)[:, 0]
# remove previous ships
STATE[mask_void, 0, 1] = 0
# remove prvious cargos
STATE[mask_void, 0, 3] = 0

# reshape stuff
STATE = STATE[:, 0, :].reshape(map_size, map_size, -1)
ACTION = ACTION[:, 0, 0].reshape(map_size, map_size)
S = S[:, 0].reshape(map_size, map_size)

if debug:
    print("0: stay still \n1:S\n2:N\n3:E\n4:W\n5:drop\n")

    print("action layer: \n")
    print("BEFORE:", action)
    print("AFTER:", ACTION, "\n")

    print("S layer: \n", S, "\n")
    print("halite layer: ")
    print("BEFORE:")
    print(mapp[:, :, 0])
    print("AFTER:")
    print(STATE[:, :, 0], "\n")

    print("cargo layer:")
    print("BEFORE:")
    print(mapp[:, :, 1])
    print("AFTER:")
    print(STATE[:, :, 1], "\n")

    print("shipy/dropoff layer: \n")
    print("BEFORE:")
    print(mapp[:, :, 2])
    print("AFTER:")
    print(STATE[:, :, 2], "\n")

    print("ship layer: \n")
    print("BEFORE:")
    print(mapp[:, :, 3])
    print("AFTER:")
    print(STATE[:, :, 3], "\n")

    print("void layer: \n")
    print("BEFORE:")
    print(mapp[:, :, 4])
    print("AFTER:")
    print(STATE[:, :, 4], "\n")

    print("void layer: \n")
    print("BEFORE:")
    print(mapp[:, :, 5])
    print("AFTER:")
    print(STATE[:, :, 5], "\n")
