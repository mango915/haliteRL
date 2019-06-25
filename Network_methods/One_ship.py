import sys
import gym

sys.path.insert(0, "../Environment/")
import halite_env as Env


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
    def __init__(self, HEnv, idd):
        self.id = idd
        self.HEnv = HEnv
        self.reward = self.Henv.player_halite[0]
        self.observation_space = HEnv.observation_space
        self.action_space = gym.spaces.Box(0, 4, shape=np.array([1]), dtype=np.int16)
        self.state = self.HEnv.map

    def step(self, action):
        action_matrix = scalar_to_matrix_action(
            action, self.state, map_size=self.state.shape[0]
        )
        state, reward, done, info = self.HEnv.step(action_matrix)
        reward = process_reward(reward)
        return state, reward, done, info

    def process_reward(self, reward):
        rew = (reward - self.reward) / 1000
        rew -= 0.1
        self.reward = reward
        return rew

    def reset():
        return HEnv.reset()


HEnv = HaliteEnv(args)
env = SingleShipEnv(HEnv, idd)
env.reset()
env.step(env.action_space.sample())
...
