use_tf = True
use_init = True
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

assert use_env == 'com'
from private_examples.com_snake_env import SnakeEnv

import joblib

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def get_session(interactive=False, mem_frac=0.1):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        gpu_options=gpu_options)
    if interactive:
        session = tf.InteractiveSession(config=tf_config)
    else:
        session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


if use_init:
    initialized_path = '/home/thanard/Downloads/rllab/data/local/snake-2-models/snake-2-models_2017_06_23_10_49_15_0001/params.pkl'
    sess = get_session(True)
    data = joblib.load(initialized_path)
    policy = data['policy']
    env = data['env']
    baseline = data['baseline']
    import numpy as np
    sess.run(tf.assign(policy._l_std_param.param, np.zeros(4)))
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=1000,
        discount=1.00,
        step_size=0.01,
    )

    algo.train()
else:
    stub(globals())
    env = normalize(SnakeEnv())
    if use_tf:
        env = TfEnv(env)
        policy = GaussianMLPPolicy(
            name='policy',
            env_spec=env.spec,
            hidden_sizes=(32,32),
            output_nonlinearity=tf.nn.tanh
        )
    else:
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32),
        )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
    )
    print(env, algo, baseline)
    run_experiment_lite(
        algo.train(),
        exp_prefix='%s_exp'%use_env,
        n_parallel = 1,
        snapshot_mode='last',
        seed=1,
    )
