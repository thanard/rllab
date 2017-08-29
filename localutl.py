#!/usr/bin/env python
import logging
import sys
import os
import click
import numpy as np
import joblib
import rllab.config as config
import subprocess
import colored_traceback
colored_traceback.add_hook(always=True)

DEBUG_LOGGING_MAP = {
    0: logging.CRITICAL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG
}
"""
Commands:
    plot
    viskit
    sim
    record
    trpo
    eval
"""
load_reset_inits_paths = {
    'reacher': 'data_upload/policy_validation_reset_inits_reacher.save',
    'snake': 'data_upload/policy_validation_reset_inits_snake.save',
    'swimmer': 'data_upload/policy_validation_inits_swimmer.save',
    'half_cheetah': "data_upload/policy_validation_reset_inits_half_cheetah.save"
}

@click.group()
@click.option('--verbose', '-v',
              help="Sets the debug noise level, specify multiple times "
                   "for more verbosity.",
              type=click.IntRange(0, 3, clamp=True),
              count=True)
@click.pass_context
def cli(ctx, verbose):
    logger_handler = logging.StreamHandler(sys.stderr)
    logger_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(logger_handler)
    logging.getLogger().setLevel(DEBUG_LOGGING_MAP.get(verbose, logging.DEBUG))

@cli.command()
@click.argument('folder_path')
def plot(folder_path):
    """ Plot MB learning curves """
    script = "sandbox/thanard/bootstrapping/plot_learning_curve.py"
    command = [
        "python",
        os.path.join(config.PROJECT_PATH, script),
        folder_path
    ]
    subprocess.check_call(command)

@cli.command()
@click.argument('path', nargs=-1)
@click.option('--port', '-p', default='5000')
def viskit(path, port):
    """ Plot multiple runs using viskit """
    script = "rllab/viskit/frontend.py"
    command = [
        "python",
        os.path.join(config.PROJECT_PATH, script),
        *path,
        "--port",
        port
    ]
    subprocess.check_call(command)

@cli.command()
@click.argument('algo')
@click.argument('env')
@click.option('-ec2', is_flag=True)
@click.option('-prefix', default=None)
@click.option('--n_seeds', '-n', default='10')
def run(algo, env, ec2, prefix, n_seeds):
    """ Run MB algo """
    script = "sandbox/thanard/bootstrapping/run_model_based_rl.py"
    command = [
        "python",
        os.path.join(config.PROJECT_PATH, script),
        algo,
        "-env",
        env,
        "-n",
        n_seeds
    ]
    if ec2:
        command.append("-ec2")
    if prefix is not None:
        command.extend(["-prefix", prefix])
    subprocess.check_call(command)

@cli.command()
@click.argument('env')
@click.option('--use_eval', '-ue', is_flag=True)
@click.option('--policy_init_path', '-ip', default=None)
@click.option('--horizon', '-h', type=str, default="100")
@click.option('--batch_size', '-n', type=str, default="4000")
@click.option('--ec2', '-ec2', is_flag=True)
@click.option('--num_exps', '-ne', type=str, default="1")
@click.option('--num_iters', '-ni', type=str, default="1000")
def trpo(env,
         use_eval,
         policy_init_path,
         horizon,
         batch_size,
         ec2,
         num_exps,
         num_iters):
    """ Run TRPO """
    script = "private_examples/run_trpo.py"
    command = [
        "python",
        os.path.join(config.PROJECT_PATH, script),
        "--env_name",
        env,
        "--horizon",
        horizon,
        "--batch_size",
        batch_size,
        "--niters",
        num_iters
    ]
    if policy_init_path is not None:
        command.extend(["--policy_init_path", policy_init_path])
    if use_eval:
        command.append("-use_eval")
    if ec2:
        command.extend(["-ec2", "--nexps", num_exps])
    subprocess.check_call(command)

@cli.command()
@click.argument('params_path')
@click.option('--horizon', '-h', type=str, default="100")
@click.option('--action_noise', '-a', type=str, default="0.0")
def sim(params_path, horizon, action_noise):
    """ Sim policy """
    script = "private_examples/sim_policy.py"
    command = [
        "python",
        os.path.join(config.PROJECT_PATH, script),
        params_path,
        "--horizon",
        horizon,
        "--action_noise",
        action_noise
    ]
    subprocess.check_call(command)

@cli.command()
@click.argument('folder')
@click.option('--horizon', '-h',
              type=str,
              default='100',
              help='Max length of rollout')
@click.option('--policy', '-p',
              type=str,
              help='path to replace default params.pkl',
              default='')
@click.option('--iter', '-i',
              help='choose iter to load from',
              type=str,
              default='final')
@click.option('--mode',
              type=str,
              default='r',
              help='choose modes')
# @click.option('--scope', '-s',
#               help="model scope",
#               type=str,
#               default='training_dynamics')
def record(folder, horizon, policy, iter, mode):
    """ Record rollouts """
    script = "private_examples/record_video.py"
    command = [
        "python",
        os.path.join(config.PROJECT_PATH, script),
        folder,
        "--horizon",
        horizon,
        "--policy",
        policy,
        "--mode",
        mode,
        "--ckpt",
        'training_logs/policy-and-models-%s.ckpt' % iter
    ]
    subprocess.check_call(command)

import pickle
import joblib
def _evaluate_fixed_inits(policy,
                          env,
                          reset_initial_states,
                          horizon):
    def f(x):
        if hasattr(env.wrapped_env, 'wrapped_env'):
            inner_env = env.wrapped_env.wrapped_env
            observation = inner_env.reset(x)
        else:
            env.reset()
            half = int(len(x) / 2)
            inner_env = env.wrapped_env.env.unwrapped
            inner_env.set_state(x[:half], x[half:])
            observation = inner_env._get_obs()
        episode_reward = 0.0
        episode_cost = 0.0
        for t in range(horizon):
            action = policy.get_action(observation)[1]['mean'][None]
            # clipping
            action = np.clip(action, *env.action_space.bounds)
            next_observation, reward, done, info = env.step(action[0])
            cost = inner_env.cost_np(observation[None], action, next_observation[None])
            # Update observation
            observation = next_observation
            # Update cost
            episode_cost += cost
            # Update reward
            episode_reward += reward
            if done:
                break
        # assert episode_cost + episode_reward < 1e-2
        return episode_cost
    # Run evaluation in parallel
    outs = np.array(list(map(f, reset_initial_states)))
    # Return avg_eps_reward and avg_eps_cost accordingly
    return np.mean(outs)

@cli.command()
@click.argument('params_path')
@click.option('--env', '-e',
              type=str)
@click.option('--horizon', '-h',
              type=int,
              default=100)
def eval(params_path, env, horizon):
    """ Eval policy """
    import tensorflow as tf
    with tf.Session() as sess:
        data = joblib.load(params_path)
        # sess.run(tf.global_variables_initializer())
        with open(os.path.join(config.PROJECT_PATH,
                               load_reset_inits_paths[env]), 'rb') as f:
            reset_inits = pickle.load(f)
        print(_evaluate_fixed_inits(data['policy'],
                                    data['env'],
                                    reset_inits,
                                    horizon
                                    ))

if __name__ == '__main__':
    cli()
