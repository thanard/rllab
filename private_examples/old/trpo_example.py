from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import gym
from private_examples.wrappers import SwimmerWrapperGym
from rllab.misc.instrument import run_experiment_lite

def run_task(*_):
    # env = normalize(SwimmerWrapperGym('Swimmer-v1'))
    env = normalize(GymEnv('Swimmer-v1'))
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32),
        learn_std = True
    )
    
    print('horizon {}'.format(env.horizon))
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=200,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()

run_experiment_lite(
    run_task,
    n_parallel = 1,
    snapshot_mode='last',
    seed=1,
)
