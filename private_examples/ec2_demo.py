import sys
from rllab.misc.instrument import run_experiment_lite
from sandbox.thanard.bootstrapping.main  import main
import rllab.misc.logger as logger
import os
from sandbox.rocky.s3.resource_manager import resource_manager
import  rllab.config as config
def test(v):
    # logger.record_tabular('tabular', 1)
    # logger.dump_tabular(with_prefix=False)
    # logger.log('Try this log')
    logger.log(logger.get_snapshot_dir())
    # logger.log(resource_manager.get_file('policy_validation_inits_swimmer_rllab.save'))
    # logger.log(os.path.join(config.PROJECT_PATH,'sandbox/thanard/bootstrapping/data/policy_validation_inits_swimmer_rllab.save'))
    # print('how about print?')
    import pickle
    dictionary = {'a':1,'b':2}

    # Loading
    filename = os.path.join(config.PROJECT_PATH, 'data_upload/policy_validation_inits_swimmer_rllab.save')
    # filename = os.path.join(logger.get_snapshot_dir(), 'sandbox/thanard/bootstrapping/data/policy_validation_inits_swimmer_rllab.save')
    # filename =  resource_manager.get_file('policy_validation_inits_swimmer_rllab.save')
    with open(filename, 'rb') as f:
        x=pickle.load(f)
    logger.log(str(x))

    # Saving
    os.makedirs(os.path.join(logger.get_snapshot_dir(),'my_dict'))
    with open(os.path.join(logger.get_snapshot_dir(),'my_dict/mydict.pkl'), 'wb') as f:
        pickle.dump(dictionary, f)

    # Loading
    with open(os.path.join(logger.get_snapshot_dir(),'my_dict/mydict.pkl'), 'rb') as f:
        x = pickle.load(f)
    logger.log(str(x))

run_experiment_lite(
    test,
    exp_prefix="my_dict",
    # Number of parallel workers for sampling
    # n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    # snapshot_mode="last",
    # use_gpu=True,
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    # seed=0,
    # mode="local",
    mode="ec2"
    # variant=dict(step_size=step_size, seed=seed),
    # plot=True,
    # terminate_machine=False,
)
# sys.exit()
