from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from private_examples.point2D_env import Point2DEnv

stub(globals())
env = normalize(Point2DEnv())
env = TfEnv(env)
policy = GaussianMLPPolicy(
    name='policy',
    env_spec=env.spec,
    hidden_sizes=(16, 16),
    # output_nonlinearity=tf.nn.tanh
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=500,
    max_path_length=10,
    n_itr=50,
    discount=1.0,
    step_size=0.1,
)

run_experiment_lite(
    algo.train(),
    exp_prefix='point2D_exp',
    n_parallel = 1,
    snapshot_mode='last',
    seed=1,
)
