from rllab.envs.gym_env import GymEnv
from rllab.envs.base import Env, Step
from rllab.envs.mujoco.swimmer_env import SwimmerEnv

# For gym
class SwimmerWrapperGym(GymEnv):
	def step(self, action):
		next_obs, reward, done, info = self.env.step(action)
		return Step(next_obs, self.env.unwrapped.model.data.qpos[0,0], done, **info)
		

class SwimmerWrapper(SwimmerEnv):
    def __init__(self, env):
        self.env = env
        super(SwimmerWrapper, self).__init__(env)

    def step(self, action):
        rllab_step = self.env.step(action)
        return Step(rllab_step.observation, rllab_step.observation[0] , rllab_step.done)