from gym.envs.registration import register

register(
    id='PuddleWorld-v0',
    entry_point='gym_puddle.envs:PuddleEnv',
)
