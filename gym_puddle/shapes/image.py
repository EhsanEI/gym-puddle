import gym

from gym.envs.classic_control import rendering

class Image(rendering.Geom):
    def __init__(self, img, width, height):
        rendering.Geom.__init__(self)
        self.width = width
        self.height = height
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(0, 0, width=self.width, height=self.height)
