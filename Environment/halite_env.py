import numpy as np
import gym


class HaliteEnv(gym.Env):
    """
    Stores the Halite III OpenAI gym environment.

    [Original, to change]
    This environment does not use Halite III's actual game engine
    (which analyzes input from terminal and is slow for RL) but instead is
    a replica in Python.

    Attributes:
    -----------
    self.map : np.ndarray
        Map of game as a 3D array. Stores different information on each "layer"
        of the array.
    Layer 0: The Halite currently on the sea floor
    Layer 1: The Halite currently on ships/factory/dropoff
    Layer 2: Whether a Factory or Dropoff exists at the layer (Factory is 1, Dropoff is -1)
    Layer 3: Whether a Ship exists at the layer
    Layer 4: Ownership
    Layer 5: Inspiration (not given as part of observation by default)

    self.mapSize : int
        Size of map (for x and y)

    self.numPlayers : int
        Number of players

    self.playerHalite : np.ndarray
        Stores the total halite a player with ownership id <index + 1> has. self.map also
        stores the total halite with the halite under factories/dropoffs, but doesn't
        include the 5000 initial.
    """

    metadata = {"render_modes": ["human"], "map_size": 0, "num_players": 0}

    def __init__(self, num_players, map_size, regen_map_on_reset=False, map_type=None):
        """
        Every environment should be derived from gym.Env and at least contain the
        variables observation_space and action_space specifying the type of possible
        observations and actions using spaces.Box or spaces.Discrete.

        Example:
        >>> EnvTest = FooEnv()
        >>> EnvTest.observation_space=spaces.Box(low=-1, high=1, shape=(3,4))
        >>> EnvTest.action_space=spaces.Discrete(2)

        HaliteEnv initialization function.
        """
        print("Initializing Halite Environment")
        self.map_generator = MapGenerator()
        self.map = self.map_generator.generate_map(map_size, num_players)
        # numPlayers = int(numPlayers)
        self.player_halite = np.empty((num_players, 1))
        self.player_halite.fill(5000)
        self.num_players = num_players
        self.map_size = map_size
        self.n_cells = map_size ** 2
        self.regen_map = regen_map_on_reset
        self.metadata["map_size"] = map_size
        self.metadata["num_players"] = num_players
        self.info = None
        self.nlayers = 7
        if not self.regen_map:
            self.original_map = self.map.copy()

    def step(self, action, makeship = False, debug=False):
        """
        Primary interface between environment and agent.

        Paramters:
            action: int
                    the index of the respective action (if action space is discrete)

        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)

        """
        # action = he.dummy_action(mapp, 10)
        rolled_sa = roll_state(self.map, action)

        directions = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
        rolled_sa = rolled_sa.reshape(5, self.n_cells, self.nlayers)
        rolled_sa = np.swapaxes(rolled_sa, 0, 1)

        state = rolled_sa[:, :, :-1]
        action = rolled_sa[:, :, -1:]

        mask_shipyard = state[:, 0, 2] == 1
        if makeship and self.player_halite[0] >= 1000: #! multyplayer TODO
            self.player_halite[0] -= 1000
            if state[mask_shipyard,0,3] == 1:
                 x = 0
            else:
                state[mask_shipyard,0,3] = 1

        # check final number of ships for every cell
        S = (directions[np.newaxis, ...] == action).sum(axis=1)

        # ACTION FIVE
        # check not a shipyard
        mask_not_shipy = state[:, 0, 2] != 1
        # check create dropoff
        mask_action_five = action[:, 0, 0] == 5

        # TODO: STATE[:,0,0] add to player's halite
        #self.player_halite[0] +=
        # check two previous checks together
        mask_five_not_shipy = np.all((mask_not_shipy, mask_action_five), axis=0)
        # remove cell's halite
        state[mask_five_not_shipy, 0, 0] = 0
        # create dropoff
        state[mask_five_not_shipy, 0, 2] = -1

        # COLLISION (S==2)
        # check cells with collision
        mask_collision = (S > 1)[:, 0]  # (S>1)[:,0] <- [:,0] is for broadcasting
        # check incoming ships
        mask_arrivals = (action == directions[np.newaxis, ...])[:, :, 0]
        # halite from neigbours where there is a collision but does not check yet if it is coming
        potential_drop = state[:, :, 1][mask_collision].copy()
        # check coming
        mask_drop = mask_arrivals[mask_collision]
        # remove cargo halite that is not coming(this is just a copy)
        potential_drop[~mask_drop] = 0
        # add dropped halite to ecean
        # TODO add dropoff case
        state[mask_collision, 0, 0] += potential_drop.sum(axis=1)
        # remove cargo halite
        state[mask_collision, 0, 1] = 0
        # remove ships
        state[mask_collision, 0, 3] = 0

        # ACTION (S==1)
        # check non interacting moves
        mask_action = (S == 1)[:, 0]  # (S>1)[:,0] <- [:,0] is for broadcasting
        # stay still move
        mask_stay = np.all((action[:, 0, 0] == 0, mask_action), axis=0)
        # calculate 25% of cell's halite
        potential_gain = np.round(state[:, 0, 0] * 0.25)
        # check actual cargos
        potential_cargos = state[:, 0, 1]
        # check fullness
        mask_not_full = (potential_cargos + potential_gain) <= 1000
        # unify stay and not full
        mask_stay_not_full = np.all((mask_stay, mask_not_full), axis=0)
        # unify stay and full
        mask_stay_full = np.all((mask_stay, ~mask_not_full), axis=0)
        # take all 25% of the halite
        state[mask_stay_not_full, 0, 0] -= potential_gain[mask_stay_not_full]
        state[mask_stay_not_full, 0, 1] += potential_gain[mask_stay_not_full]
        # take halite only for fill the space left
        space_left = 1000 - state[mask_stay_full, 0, 1]
        state[mask_stay_full, 0, 0] -= space_left
        state[mask_stay_full, 0, 1] = 1000

        # movement step
        mask_coming_ships = np.squeeze((directions[np.newaxis, ...] == action), axis=2)[
            mask_action
        ]
        # ship arrive
        state[mask_action, 0, 3] = 1
        # cargo arrive
        mask_dropoff = state[:, 0, 2] == -1
        state[mask_action, 0, 1] = state[:, :, 1][mask_action][mask_coming_ships]
        self.player_halite[0] += state[mask_shipyard or mask_dropoff, 0, 1].sum()
        state[mask_shipyard or mask_dropoff, 0, 1] = 0

        # VOID (S==0)
        # check no ships in cell
        mask_void = (S == 0)[:, 0]
        # remove previous ships
        state[mask_void, 0, 1] = 0
        # remove prvious cargos
        state[mask_void, 0, 3] = 0

        # reshape stuff
        state = state[:, 0, :].reshape(self.map_size, self.map_size, -1)
        if debug:
            action = action[:, 0, 0].reshape(self.map_size, self.map_size)
            S = S[:, 0].reshape(self.map_size, self.map_size)
            print("0: stay still \n1:S\n2:N\n3:E\n4:W\n5:drop\n")

            print("action layer: \n")
            print("BEFORE:", action)
            print("AFTER:", action, "\n")

            print("S layer: \n", S, "\n")
            print("halite layer: ")
            print("BEFORE:")
            print(self.map[:, :, 0])
            print("AFTER:")
            print(state[:, :, 0], "\n")

            print("cargo layer:")
            print("BEFORE:")
            print(self.map[:, :, 1])
            print("AFTER:")
            print(state[:, :, 1], "\n")

            print("shipy/dropoff layer: \n")
            print("BEFORE:")
            print(self.map[:, :, 2])
            print("AFTER:")
            print(state[:, :, 2], "\n")

            print("ship layer: \n")
            print("BEFORE:")
            print(self.map[:, :, 3])
            print("AFTER:")
            print(state[:, :, 3], "\n")

            print("void layer: \n")
            print("BEFORE:")
            print(self.map[:, :, 4])
            print("AFTER:")
            print(state[:, :, 4], "\n")

            print("void layer: \n")
            print("BEFORE:")
            print(self.map[:, :, 5])
            print("AFTER:")
            print(state[:, :, 5], "\n")
            return state, self.mapp, action
        self.map = state
        return state

    def reset(self):
        """
        This method resets the environment to its initial values.

        Returns:
            observation:    array
                            the initial state of the environment
        """
        if not self.regen_map:
            self.map = self.original_map.copy()
        else:
            self.map = self.map_generator.generate_map(self.map_size, self.num_players)
        self.player_halite = np.empty((self.num_players, 1))
        self.player_halite.fill(5000)
        return self.map

    def render(self, mode="human", close=False):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        pass


def MapSize(size):
    """
    Enum of the different possible map sizes
    """
    defaults = dict(TINY=32, SMALL=40, MEDIUM=48, LARGE=56, GIANT=64)
    if size in defaults:
        return defaults[size]
    else:
        return size


class MapGenerator:
    def __init__(self):
        return

    def generate_fractal_map(self, map_size, num_players):
        return np.zeros(2)

    def generate_map(self, map_size, num_players):
        if num_players != 1 and num_players != 2 and num_players != 4:
            raise Exception("Only 1, 2 or 4 players are supported")

        shape = (map_size, map_size)
        layer = np.zeros(shape, dtype=np.int64)
        mapp = np.tile(layer[:, :, np.newaxis], 4)  # 6)

        # halite layer
        mapp[:, :, 0] = np.random.randint(1e4, size=shape)

        # halite on ships layer (nothing to change)
        mapp[:, :, 1] = 0

        # shipyard, dropoff location (+1 shipyards, -1 dropoffs)
        self.initialize_shipyard_location(map_size, num_players, mapp)
        # remove halite under shipyard starting position
        mapp[:, :, 0][mapp[:, :, 2] == 1] = 0

        # ships locations (nothing to change)
        mapp[:, :, 3] = 0

        # nothing for now
        # ship and buildings ownership (nothing to change)
        # mapp[:, :, 4]

        # ispiration (nothing to change)
        # mapp[:, :, 5]

        return mapp

    def initialize_shipyard_location(self, map_size, num_players, mapp):
        if num_players == 1:
            x = y = map_size // 2
            mapp[x, y, [2, 4]] = 1
        elif num_players == 2:
            x1 = map_size // 4
            x2 = map_size - (map_size // 4)
            y1 = y2 = map_size // 2
            mapp[x1, y1, [2, 4]] = 1
            mapp[x2, y2, [2, 4]] = 1
        elif num_players == 4:
            x1 = x3 = map_size // 4
            x2 = x4 = map_size - (map_size // 4)
            y1 = y2 = map_size // 4
            y3 = y4 = map_size - (map_size // 4)
            mapp[x1, y1, [2, 4]] = 1
            mapp[x2, y2, [2, 4]] = 1
            mapp[x3, y3, [2, 4]] = 1
            mapp[x4, y4, [2, 4]] = 1

    def dummy_map(self, map_size, num_ships):
        mapp = self.generate_map(map_size, 1).reshape(map_size ** 2, 6)
        ships_locations = np.random.choice(map_size ** 2, size=num_ships, replace=False)
        mapp[ships_locations, 3] = 1
        mapp[ships_locations, 1] = np.random.randint(1000, size=num_ships)
        return mapp.reshape(map_size, map_size, 6)


def dummy_action(state, num_ships):
    action = np.zeros((state.shape[0], state.shape[1])) - 1
    action[state[:, :, 3].astype(np.bool)] = np.random.choice(6, size=num_ships)
    return action


def roll_state(state, action):
    SA = np.concatenate((state, action[..., np.newaxis]), axis=2)
    SAn = np.roll(SA[:, :, :], shift=1, axis=0)
    SAs = np.roll(SA[:, :, :], shift=-1, axis=0)
    SAw = np.roll(SA[:, :, :], shift=1, axis=1)
    SAe = np.roll(SA[:, :, :], shift=-1, axis=1)
    rolledSA = np.stack((SA, SAn, SAs, SAw, SAe))
    return rolledSA


# def unroll_state(STATE):
#    return     STATE[:,0,:]
