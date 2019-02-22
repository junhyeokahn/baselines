import os
import time
import numpy as np
import os.path as osp
from tqdm import tqdm
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.ppo2.model import Model
from baselines.common.tf_util import get_session
from ruamel.yaml import YAML

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.process_runner import ProcessRunner

import tensorflow as tf

def constfn(val):
    def f(_):
        return val
    return f

def learn(env, nenvs, network, password, total_timesteps=1e6, seed=None,
        nsteps=2048, ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5,
        gamma=0.99, lam=0.95, log_interval=10, nminibatches=4, noptepochs=4,
        cliprange=0.2, save_interval=0, save_path=None, load_path=None, **network_kwargs):

    set_global_seeds(seed)
    save_dir = save_path

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, value_network='copy', **network_kwargs)

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                  nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm)

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = ProcessRunner(env=env, model=model, n_env=nenvs, n_steps=nsteps,
            gamma=gamma, lam=lam, password = password, verbose=0,
            **network_kwargs)

    epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    if save_interval is None:
        save_interval = nupdates // 5

    for update in range(1, nupdates+1):
        logger.log("# " + "="*78)
        logger.log("# Iteration %i / %i" % (update, nupdates))
        logger.log("# " + "="*78)
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        policy_param = get_session().run(tf.trainable_variables('ppo2_model/pi'))
        valfn_param = get_session().run(tf.trainable_variables('ppo2_model/vf'))
        obs, rewards, returns, masks, actions, values, neglogpacs, action_mean, states, epinfos, dataset_total_rew = runner.run(policy_param, valfn_param) #pylint: disable=E0632
        ## !! TEST !!
        # with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # da_, v_, nglp_, mean_, std_, logstd_ = policy().step_debug(obs, actions)
            # if not ((np.isclose(da_, action_mean, atol=5e-7)).all()):
                # print(da_ - action_mean)
                # print("action no match")
            # if not ((np.isclose(v_, values, atol=5e-7)).all()):
                # print(v_ - values)
                # print("value no match")
            # if not ((np.isclose(nglp_, neglogpacs, atol=5e-7)).all()):
                # print(nglp_-neglogpacs)
                # print("neglogp no match")
            # __import__('ipdb').set_trace()
            # print("Debugging!")
        ## !! TEST !!

        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('dataset_rew', dataset_total_rew/nenvs)
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('serial_num_dones', int(masks.sum()/nenvs))
            logger.logkv('total_num_dones', masks.sum())
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if (update == nupdates and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)) \
                or \
                (save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            save_path = osp.join(checkdir, '%.5i'%update)
            print('Saving TF model to', save_path)
            model.save(save_path)
            save_dataset(save_path, nsteps, obs, rewards, returns, masks, actions, values)
            save_model_to_yaml(save_path, **network_kwargs)

    ## !! TEST !! ##
    # with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
        # __import__('ipdb').set_trace()
        # while(True):
            # da_, v_, nglp_, mean_, std_, logstd_ = policy().step_debug(obs, actions)
    ## !! TEST !! ##

    return model

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def save_dataset(save_path, nsteps, obs, rew, ret, mask, action, value):
    n_steps = np.array([nsteps])
    np.savez(save_path+'.npz', n_steps=n_steps, obs=obs, rew=rew, ret=ret, mask=mask, action=action,
            value=value)

def save_model_to_yaml(save_path, **network_kwargs):

    _policy_param = get_session().run(tf.trainable_variables('ppo2_model/pi'))
    _valfn_param = get_session().run(tf.trainable_variables('ppo2_model/vf'))

    p_num_layer = (int) ((len(_policy_param)-1) / 2)
    v_num_layer = (int) (len(_valfn_param) / 2)
    assert(p_num_layer == v_num_layer)
    assert(p_num_layer == network_kwargs['num_layers'] + 1)
    act_fn = network_kwargs['activation']
    if act_fn == None:
        act_fn_ = 0
    elif act_fn == tf.tanh:
        act_fn_ = 1
    elif act_fn == tf.nn.relu:
        act_fn_ = 2
    else:
        print("Wrong activation function")

    pol_params = {}
    pol_params['num_layer'] = p_num_layer
    for layer_idx in range(p_num_layer):
        pol_params['w'+str(layer_idx)] = _policy_param[2*layer_idx].tolist()
        pol_params['b'+str(layer_idx)] = (_policy_param[2*layer_idx+1]).reshape(1, (_policy_param[2*layer_idx+1]).shape[0]).tolist()
        if (layer_idx == (p_num_layer - 1)):
            pol_params['act_fn'+str(layer_idx)] = 0
        else:
            pol_params['act_fn'+str(layer_idx)] = act_fn_
    pol_params['logstd'] = _policy_param[-1].tolist()

    valfn_params = {}
    valfn_params['num_layer'] = v_num_layer
    for layer_idx in range(p_num_layer):
        valfn_params['w'+str(layer_idx)] = _valfn_param[2*layer_idx].tolist()
        valfn_params['b'+str(layer_idx)] = _valfn_param[2*layer_idx+1].reshape(1, (_valfn_param[2*layer_idx+1]).shape[0]).tolist()
        if (layer_idx == (v_num_layer - 1)):
            valfn_params['act_fn'+str(layer_idx)] = 0
        else:
            valfn_params['act_fn'+str(layer_idx)] = act_fn_

    data = {"pol_params": pol_params, "valfn_params": valfn_params}
    with open(save_path + '.yaml', 'w') as f:
        yaml = YAML()
        yaml.dump(data, f)

    print('Saving Yaml to', save_path)
