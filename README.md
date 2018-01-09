# gym-puddle
Puddle world environment for OpenAI Gym with continuous state space and discrete action space.

<kbd>
  <img src='screenshot.png'/>
</kbd>

## Installation
`pip3 install -e .`

## Usage
```python
import gym
import gym_puddle # Don't forget this extra line!

env = gym.make('PuddleWorld-v0')
```

##  Notes
Rendering is available, although the program runs faster without it.
Run this line after each time step to see the puddles and agent location.
```python
env.render()
```

## References

Off-Policy Actor-Critic. Thomas Degris, Martha White, Richard S. Sutton. In *Proceedings of the Twenty-Ninth International Conference on Machine Learning (ICML)*, 2012.

http://rlpark.github.io/ (Java implementation)

https://github.com/samindaa/RLLib (C++ implementation)
