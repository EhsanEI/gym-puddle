import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy

class PuddleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

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
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self.pos += self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = self._get_reward(self.pos)

        done = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold

        return self.pos, reward, done, {}

    def _get_reward(self, pos):
        reward = -1.
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            reward -= 2. * self._gaussian1d(pos[0], cen[0], wid[0]) * \
                self._gaussian1d(pos[1], cen[1], wid[1])

        return reward

    def _gaussian1d(self, p, mu, sig):
        return np.exp(-((p - mu)**2)/(2.*sig**2)) / (sig*np.sqrt(2.*np.pi))

    def _reset(self):
        if self.start is None:
            self.pos = self.observation_space.sample()
        else:
            self.pos = copy.copy(self.start)
        return self.pos

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            from gym_puddle.shapes.image import Image
            self.viewer = rendering.Viewer(screen_width, screen_height)

            import pyglet
            img_width = 100
            img_height = 100
            fformat = 'RGB'
            pixels = np.zeros((img_width, img_height, len(fformat)))
            for i in range(img_width):
                for j in range(img_height):
                    x = float(i)/img_width
                    y = float(j)/img_height
                    pixels[j,i,:] = self._get_reward(np.array([x,y]))

            pixels -= pixels.min()
            pixels *= 255./pixels.max()
            pixels = np.floor(pixels)

            img = pyglet.image.create(img_width, img_height)
            img.format = fformat
            data=[chr(int(pixel)) for pixel in pixels.flatten()]

            img.set_data(fformat, img_width * len(fformat), ''.join(data))
            bg_image = Image(img, screen_width, screen_height)
            bg_image.set_color(1.0,1.0,1.0)

            self.viewer.add_geom(bg_image)

            thickness = 5
            agent_polygon = rendering.FilledPolygon([(-thickness,-thickness),
             (-thickness,thickness), (thickness,thickness), (thickness,-thickness)])
            agent_polygon.set_color(0.0,1.0,0.0)
            self.agenttrans = rendering.Transform()
            agent_polygon.add_attr(self.agenttrans)
            self.viewer.add_geom(agent_polygon)

        self.agenttrans.set_translation(self.pos[0]*screen_width, self.pos[1]*screen_height)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
