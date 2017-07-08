from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from private_examples.point_mass_env import PointMassEnv

stub(globals())
env = normalize(PointMassEnv())
env = TfEnv(env)
policy = GaussianMLPPolicy(
    name='policy',
    env_spec=env.spec,
    hidden_sizes=(32, 32),
    # output_nonlinearity=tf.nn.tanh
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

kwargs = dict(
    reset_init_path="data_upload/policy_validation_inits_point_mass_rllab.save",
    horizon=50
)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=50,
    n_itr=1000,
    discount=1.0,
    step_size=0.01,
    **kwargs
)
run_experiment_lite(
    algo.train(),
    exp_prefix='point_mass_exp',
    n_parallel = 1,
    snapshot_mode='last',
    seed=1,
)