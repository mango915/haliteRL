import numpy as np
import gym
import random


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
    Layer 1: Whether a Ship exists at the layer
    Layer 2: The Halite currently on ships/factory/dropoff
    Layer 3: Whether a Factory or Dropoff exists at the layer (Factory is 1, Dropoff is -1)
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

    def __init__(
        self,
        num_players,
        map_size,
        episode_lenght=400,
        regen_map_on_reset=False,
        map_type=None,
        id_max=100,
        verbosity=0,
    ):
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

        self.verbosity = verbosity
        if self.verbosity > 0:
            print("Initializing Halite Environment")

        # Map variables
        self.map_generator = MapGenerator()
        self.map = self.map_generator.generate_map(map_size, num_players)
        self.map_size = map_size
        self.n_cells = map_size ** 2
        self.regen_map = regen_map_on_reset
        self.metadata["map_size"] = map_size
        self.nlayers = 6
        if not self.regen_map:
            self.original_map = self.map.copy()

        # Players variables
        self.player_halite = np.empty(num_players)
        self.player_halite.fill(5000)
        self.num_players = num_players
        self.metadata["num_players"] = num_players

        # Gym consistency
        self.info = {}

        # Turn metadata
        self.turn = 0
        self.endturn = episode_lenght

        # Ships variables
        self.id = 1  # starting shipd id
        self.id_max = 100
        self.ships = []

        # Gym spaces
        obs_shape = (map_size, map_size, self.nlayers - 1)
        low_obs = np.zeros(obs_shape, dtype=np.int16)
        high_obs = np.zeros(obs_shape, dtype=np.int16)
        # halite and cargo layers
        high_obs[:, :, [0, 2]] = 1000
        # ships layer
        high_obs[:, :, 1] = 1
        # shipyard and dropoff layer
        high_obs[:, :, 3] = 1
        low_obs[:, :, 3] = -1
        # !!!WARNING: if there is more than 100 ships you have to change this line
        # id layer
        high_obs[:, :, 4] = id_max

        self.observation_space = gym.spaces.Box(low_obs, high_obs, dtype=np.int16)
        self.action_space = gym.spaces.Box(
            0, 5, shape=(map_size, map_size), dtype=np.int16
        )

    def step(self, action, makeship=False, debug=False):
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

        # check final number of ships for every cell
        S = (directions[np.newaxis, ...] == action).sum(axis=1)

        mask_shipyard = state[:, 0, 3] == 1
        if makeship and self.player_halite[0] >= 1000:  #! multyplayer TODO
            # TODO check if it work
            self.player_halite[0] -= 1000
            if S[mask_shipyard, 0][0] > 0:
                S[mask_shipyard, 0] += 1
            else:
                state[mask_shipyard, 0, 1] = 1
                state[mask_shipyard, 0, -1] = self.id
                self.id += 1
                assert (
                    self.id <= self.id_max
                ), "You are making too much ships! if you want to do more initialize the environment with a bigger 'id_max' parameter"
                S[mask_shipyard, 0] = 1
                action[mask_shipyard, 0, 0] = 0

        # ACTION FIVE
        # check not a shipyard
        mask_not_shipy = state[:, 0, 3] != 1
        # check create dropoff
        mask_action_five = action[:, 0, 0] == 5

        # TODO: STATE[:,0,0] add to player's halite
        # self.player_halite[0] +=
        # check two previous checks together
        mask_five_not_shipy = np.all((mask_not_shipy, mask_action_five), axis=0)
        # remove cell's halite
        state[mask_five_not_shipy, 0, 0] = 0
        # create dropoff
        state[mask_five_not_shipy, 0, 3] = -1

        # COLLISION (S==2)
        # check cells with collision
        mask_collision = (S > 1)[:, 0]  # (S>1)[:,0] <- [:,0] is for broadcasting
        # check incoming ships
        mask_arrivals = (action == directions[np.newaxis, ...])[:, :, 0]
        # halite from neigbours where there is a collision but does not check yet if it is coming
        potential_drop = state[:, :, 2][mask_collision].copy()
        # check coming
        mask_drop = mask_arrivals[mask_collision]
        # remove cargo halite that is not coming(this is just a copy)
        potential_drop[~mask_drop] = 0
        # add dropped halite to ecean
        # TODO add dropoff case
        state[mask_collision, 0, 0] += potential_drop.sum(axis=1)
        # remove cargo halite
        state[mask_collision, 0, 2] = 0
        # remove ships
        state[mask_collision, 0, 1] = 0
        state[mask_collision, 0, -1] = 0

        # ACTION (S==1)
        # check non interacting moves
        mask_action = (S == 1)[:, 0]  # (S>1)[:,0] <- [:,0] is for broadcasting
        # stay still move
        mask_stay = np.all((action[:, 0, 0] == 0, mask_action), axis=0)
        # calculate 25% of cell's halite
        potential_gain = np.round(state[:, 0, 0] * 0.25).astype("int64")
        # check actual cargos
        potential_cargos = state[:, 0, 2]
        # check fullness
        mask_not_full = (potential_cargos + potential_gain) <= 1000
        # unify stay and not full
        mask_stay_not_full = np.all((mask_stay, mask_not_full), axis=0)
        # unify stay and full
        mask_stay_full = np.all((mask_stay, ~mask_not_full), axis=0)
        # take all 25% of the halite
        state[mask_stay_not_full, 0, 0] -= potential_gain[mask_stay_not_full]
        state[mask_stay_not_full, 0, 2] += potential_gain[mask_stay_not_full]
        # take halite only for fill the space left
        space_left = 1000 - state[mask_stay_full, 0, 2]
        state[mask_stay_full, 0, 0] -= space_left
        state[mask_stay_full, 0, 2] = 1000

        # movement step
        mask_coming_ships = np.squeeze((directions[np.newaxis, ...] == action), axis=2)[
            mask_action
        ]
        # ship arrive
        state[mask_action, 0, 1] = 1
        state[mask_action, 0, -1] = state[:, :, -1][mask_action][mask_coming_ships]
        # cargo arrive
        mask_dropoff = state[:, 0, 3] == -1
        state[mask_action, 0, 2] = state[:, :, 2][mask_action][mask_coming_ships]
        self.player_halite[0] += state[
            np.any((mask_shipyard, mask_dropoff), axis=0), 0, 2
        ].sum()
        state[np.any((mask_shipyard, mask_dropoff), axis=0), 0, 2] = 0

        # VOID (S==0)
        # check no ships in cell
        mask_void = (S == 0)[:, 0]
        # remove previous ships
        state[mask_void, 0, 1] = 0
        state[mask_void, 0, -1] = 0
        # remove prvious cargos
        state[mask_void, 0, 2] = 0

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
            print(self.map[:, :, 2])
            print("AFTER:")
            print(state[:, :, 2], "\n")

            print("shipy/dropoff layer: \n")
            print("BEFORE:")
            print(self.map[:, :, 3])
            print("AFTER:")
            print(state[:, :, 3], "\n")

            print("ship layer: \n")
            print("BEFORE:")
            print(self.map[:, :, 1])
            print("AFTER:")
            print(state[:, :, 1], "\n")

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
        self.turn += 1
        state = state.astype(np.int64)
        if self.turn >= self.endturn:
            return state, self.player_halite[0], True, self.info
        else:
            return state, self.player_halite[0], False, self.info

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
        self.id = 1
        self.turn = 0
        return self.map

    def render(self, mode="human", close=False):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        print("⛵️")
        print("⚓")
        print("🏰")

        pass

    def seed(self, seed):
        random.seed(seed)


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
        mapp = np.tile(layer[:, :, np.newaxis], 5)  # 6)

        # halite layer
        mapp[:, :, 0] = np.random.randint(1e3, size=shape)

        # halite on ships layer (nothing to change)
        mapp[:, :, 2] = 0
        #!!!!!!!!!!!!!!!!!!! 3->1, 1->2, 2->3
        # shipyard, dropoff location (+1 shipyards, -1 dropoffs)
        self.initialize_shipyard_location(map_size, num_players, mapp)
        # remove halite under shipyard starting position
        mapp[:, :, 0][mapp[:, :, 3] == 1] = 0

        # ships locations (nothing to change)
        mapp[:, :, 1] = 0

        # ship id
        mapp[:, :, 4] = 0

        # ship and buildings ownership (nothing to change)
        # mapp[:, :, 5]
        # nothing for now

        # ispiration (nothing to change)
        # mapp[:, :, 6]

        return mapp

    def initialize_shipyard_location(self, map_size, num_players, mapp):
        if num_players == 1:
            x = y = map_size // 2
            mapp[x, y, 3] = 1
        elif num_players == 2:
            x1 = map_size // 4
            x2 = map_size - (map_size // 4)
            y1 = y2 = map_size // 2
            mapp[x1, y1, [3, 4]] = 1
            mapp[x2, y2, [3, 4]] = 1
        elif num_players == 4:
            x1 = x3 = map_size // 4
            x2 = x4 = map_size - (map_size // 4)
            y1 = y2 = map_size // 4
            y3 = y4 = map_size - (map_size // 4)
            mapp[x1, y1, [3, 4]] = 1
            mapp[x2, y2, [3, 4]] = 1
            mapp[x3, y3, [3, 4]] = 1
            mapp[x4, y4, [3, 4]] = 1

    def dummy_map(self, map_size, num_ships):
        mapp = self.generate_map(map_size, 1).reshape(map_size ** 2, 6)
        ships_locations = np.random.choice(map_size ** 2, size=num_ships, replace=False)
        mapp[ships_locations, 1] = 1
        mapp[ships_locations, 2] = np.random.randint(1000, size=num_ships)
        return mapp.reshape(map_size, map_size, 6)


def dummy_action(state, num_ships):
    action = np.zeros((state.shape[0], state.shape[1])) - 1
    action[state[:, :, 1].astype(np.bool)] = np.random.choice(6, size=num_ships)
    return action


def roll_state(state, action):
    SA = np.concatenate((state, action[..., np.newaxis]), axis=2)
    SAn = np.roll(SA[:, :, :], shift=1, axis=0)
    SAs = np.roll(SA[:, :, :], shift=-1, axis=0)
    SAw = np.roll(SA[:, :, :], shift=1, axis=1)
    SAe = np.roll(SA[:, :, :], shift=-1, axis=1)
    rolledSA = np.stack((SA, SAn, SAs, SAw, SAe))
    return rolledSA


def one_to_index(V, L):
    # matrix V with one entry = 1 and the others 0
    return np.arange(L ** 2).reshape((L, L))[V.astype(bool)]


def decode(v_enc, L):
    V = np.arange(0, L ** 2).reshape((L, L))
    v_dec = np.array([np.where(v_enc == V)[0][0], np.where(v_enc == V)[1][0]])
    return v_dec


def scalar_to_matrix_action(action, state, idd, map_size=7):
    # first get the decoded position of the ship
    ship_pos_matrix = state[:, :, 4]
    ship_pos_matrix[state[:, :, 4] != idd] = 0
    pos_enc = one_to_index(ship_pos_matrix, map_size)
    pos_dec = decode(pos_enc, map_size)
    # then fill a matrix of -1
    mat_action = np.full((map_size, map_size), -1)
    # finally insert the action in the pos_dec entry
    mat_action[tuple(pos_dec)] = action
    return mat_action


class SingleShipEnv(gym.Env):
    def __init__(self, HEnv, idd, map_size):
        self.id = idd
        self.HEnv = HEnv
        self.map_size = map_size
        self.reward = self.HEnv.player_halite[0]
        self.cargo = 0
        self.observation_space = HEnv.observation_space
        self.action_space = gym.spaces.Discrete(4)
        self.state = self.HEnv.map

    def step(self, action):
        action_matrix = scalar_to_matrix_action(
            action, self.state, self.id, map_size=self.state.shape[0]
        )
        state, reward, done, info = self.HEnv.step(action_matrix)
        cargo = state[state[:, :, 4] == self.id, 2]
        # print(cargo, done, reward)
        reward = self.process_reward(reward, cargo)
        self.state = state
        return state, reward, done, info

    def process_reward(self, reward, cargo):
        rew = (reward - self.reward) / 1000
        rew -= 0.01
        rew += (cargo - self.cargo) / 5000
        self.cargo = cargo
        if int(self.cargo) == 1000:
            rew -= 0.2
        self.reward = reward
        return rew[0]

    def seed(self, seed):
        random.seed(seed)
        self.HEnv.seed(seed)
        
    def render(self, mode="human", close=False):
        self.HEnv.render(mode, close)

    def reset(self):
        self.HEnv.reset()
        action = np.zeros((self.map_size, self.map_size)) - 1
        state, rew, done, info = self.HEnv.step(action, makeship=True)
        self.state = state
        return self.state

    def pl_halite(self):
        return self.HEnv.player_halite[0]