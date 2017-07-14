use_tf = True
use_init = False
use_env = 'local'
if use_tf:
    import tensorflow as tf
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.envs.base import TfEnv
else:
    from rllab.algos.trpo import TRPO
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
if use_env == 'gym':
    from rllab.envs.gym_env import GymEnv
elif use_env == 'local':
    from rllab.envs.gym_env import GymEnv
    from private_examples.reacher_env import ReacherEnv
else:
    assert False

initialized_path = ""
kwargs = dict(
    reset_init_path="data_upload/policy_validation_reset_inits_reacher.save",
    horizon=50
)

stub(globals())
env = TfEnv(GymEnv("Reacher-v1", record_video=False, record_log=False))
if use_env == 'local':
    env.wrapped_env.env.env = ReacherEnv()
policy = GaussianMLPPolicy(
    name='policy',
    env_spec=env.spec,
    hidden_sizes=(32, 32),
    # output_nonlinearity=tf.nn.tanh
)
baseline = LinearFeatureBaseline(env_spec=env.spec)

if use_init:
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        initialized_path=initialized_path
    )
else:
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=50,
        n_itr=500,
        discount=1.00,
        step_size=0.01,
        **kwargs
)

run_experiment_lite(
    algo.train(),
    exp_prefix='reacher_%s_exp'%use_env,
    n_parallel = 1,
    snapshot_mode='last',
    seed=1
)

# import rllab.config as config
# def get_aws_config(count):
#     region = "us-west-1"
#     zone = "us-west-1b"
#     # if count %3 == 0:
#     #     region = "us-east-1"
#     #     zone = "us-east-1a"
#     # elif count%3 == 1:
#     #     region = "us-west-1"
#     #     zone = "us-west-1b"
#     # else:
#     #     region = "us-west-2"
#     #     zone = "us-west-2b"
#     config.AWS_REGION_NAME = region
#     aws_config = dict(
#         image_id=config.ALL_REGION_AWS_IMAGE_IDS[region],
#         key_name=config.ALL_REGION_AWS_KEY_NAMES[region],
#         network_interfaces=[
#             dict(
#                 SubnetId=config.ALL_SUBNET_INFO[zone]["SubnetID"],
#                 Groups=[config.ALL_SUBNET_INFO[zone]["Groups"]],
#                 DeviceIndex=0,
#                 AssociatePublicIpAddress=True,
#             )
#         ]
#     )
#     return aws_config
# for seed in range(20):
#     run_experiment_lite(
#         algo.train(),
#         exp_prefix='reacher_%s_exp'%use_env,
#         n_parallel = 1,
#         snapshot_mode='last',
#         mode='ec2',
#         seed=seed,
#         aws_config=get_aws_config(1)
#     )
