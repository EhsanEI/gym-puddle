import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class PuddleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, start=[0.2, 0.4], goal=[1.0, 1.0], goal_threshold=0.1,
            noise=0.025, thrust=0.05, puddle_center=[[.3, .6], [.4, .5], [.8, .9]],
            puddle_width=[[.1, .03], [.03, .1], [.03, .1]]):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.puddle_center = [np.array(center) for center in puddle_center]
        self.puddle_width = [np.array(width) for width in puddle_width]

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.actions = [np.zeros(2) for i in range(5)]
        for i in range(4):
            self.actions[i][i//2] = thrust * (i%2 * 2 - 1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self.pos += self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = -1.
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            reward -= 2. * self._gaussian1d(self.pos[0], cen[0], wid[0]) * \
                self._gaussian1d(self.pos[1], cen[1], wid[1])

        done = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold

        return self.pos, reward, done, {}

    def _gaussian1d(self, p, mu, sig):
        return np.exp(-((p - mu)**2)/(2.*sig**2)) / (sig*np.sqrt(2.*np.pi))

    def _reset(self):
        if self.start is None:
            self.pos = self.observation_space.sample()
        else:
            self.pos = self.start
        return self.pos

    def _render(self, mode='human', close=False):
        return
