use_tf = True
use_init = False
use_env = 'com'
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
if use_env == 'rllab':
    from rllab.envs.mujoco.swimmer_env import SwimmerEnv
elif use_env == 'augmented':
    from private_examples.swimmer_local_env import SwimmerEnv
else:
    assert use_env == 'com'
    from private_examples.com_swimmer_env import SwimmerEnv

# 1.0 TRPO
# initialized_path = 'data/local/experiment/experiment_2017_04_10_12_02_39_0001/params.pkl'
# 0.3 TRPO with tanh output
# initialized_path = '/home/thanard/Dropbox/UC Berkeley/Research/bootstrapping/data/params-0.3-initialized-trpo.pkl'
# 0.3 TRPO without output nonlinearity
# initialized_path = '/home/thanard/Dropbox/UC Berkeley/Research/bootstrapping/data/params-0.3-initialized-trpo-no-action-squashing.pkl'
#     initialized_path = '/home/thanard/Downloads/rllab/data/s3/bptt-see-if-successful/bptt-see-if-successful_2017_06_13_20_19_34_0017/params.pkl'
initialized_path = '/home/thanard/Downloads/rllab/data/s3/swimmer-512-model/swimmer-512-model_2017_07_02_01_32_23_0015/params.pkl'
# kwargs = dict(
#     reset_init_path="data_upload/policy_validation_inits_swimmer_rllab.save",
#     cost_np=cost_np,
#     horizon=100
# )

stub(globals())
env = normalize(SwimmerEnv())
if use_tf:
    env = TfEnv(env)
    policy = GaussianMLPPolicy(
        name='policy',
        env_spec=env.spec,
        hidden_sizes=(32,32),
        # output_nonlinearity=tf.nn.tanh
    )
else:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32),
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
        max_path_length=100,
        n_itr=500,
        discount=1.00,
        step_size=0.01
    )
run_experiment_lite(
    algo.train(),
    exp_prefix='%s_exp'%use_env,
    n_parallel = 1,
    snapshot_mode='last',
    seed=1,
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
#         exp_prefix='%s_exp' % use_env,
#         n_parallel=1,
#         snapshot_mode='last',
#         seed=seed,
#         mode="ec2",
#         variant=dict(seed=seed),
#         aws_config=get_aws_config(1)
#     )