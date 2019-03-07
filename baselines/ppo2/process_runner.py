import tensorflow as tf
from baselines.common.process_manager import ProcessManager

import numpy as np
import yaml
import zmq
import os
from tqdm import tqdm

from rl_msg_pb2 import *

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class ProcessRunner(object):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, env, model, n_env, n_steps, gamma, lam, password,
            verbose=0, **network_kwargs):
        self.env = env
        # assume env spec looks like DummyNAMEEnv-v0
        self.env_name = (env.unwrapped.spec.id.split('-')[0][5:])[:-3]
        self.model = model
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.lam = lam
        self.gamma = gamma
        self.n_env = n_env
        self.n_steps = n_steps
        self.verbose = verbose
        self.n_layer = network_kwargs['num_layers']
        self.n_hidden = network_kwargs['num_hidden']
        act_fn = network_kwargs['activation']
        if act_fn == None:
            self.act_fn = NeuralNetworkParam.NONE
        elif act_fn == tf.tanh:
            self.act_fn = NeuralNetworkParam.Tanh
        elif act_fn == tf.nn.relu:
            self.act_fn = NeuralNetworkParam.ReLU
        else:
            print("Wrong activation function")
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()

        self.parameter_setting()
        self.process_manager_list = { str(env_id): ProcessManager(self.ip_control_pc,
                                                   self.username, password,
                                                   self.execute_cmd+str(env_id), self.exit_cmd+str(env_id) + "'",
                                                   self.verbose) for env_id in range(self.n_env) }
        if self.verbose >= 1:
            print("[[Process Manager created]]")

        self.context_list = { str(env_id): zmq.Context.instance() for env_id in range(self.n_env)}
        self.data_socket_list = { str(env_idx):None for env_idx in range(self.n_env) }
        self.policy_valfn_socket_list = { str(env_idx):None for env_idx in range(self.n_env) }
        if self.verbose >= 1:
            print("[[Context created]]")

    def create_zmq_sockets(self, env_idx):
        self.data_socket_list[str(env_idx)] = self.context_list[str(env_idx)].socket(zmq.SUB)
        self.data_socket_list[str(env_idx)].setsockopt_string(zmq.SUBSCRIBE, "")
        self.data_socket_list[str(env_idx)].connect(self.ip_sub_pub_list[str(env_idx)])
        self.policy_valfn_socket_list[str(env_idx)] = self.context_list[str(env_idx)].socket(zmq.REQ)
        self.policy_valfn_socket_list[str(env_idx)].connect(self.ip_req_rep_list[str(env_idx)])
        if self.verbose >= 1:
            print("[[Socket created for %d th Env]]" % env_idx)

    def parameter_setting(self):
        cfg_path = os.getcwd() + '/Config/' + self.env_name + '/TEST/RL_WALKING_TEST.yaml'
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
            ip_sub_pub_first = config['test_configuration']['protocol']['ip_sub_pub_prefix']
            ip_req_rep_first = config['test_configuration']['protocol']['ip_req_rep_prefix']
            self.username = config['test_configuration']['protocol']['username']
            self.ip_control_pc = config['test_configuration']['protocol']['ip_control_pc']
            self.execute_cmd = config['test_configuration']['protocol']['execute_cmd']
            self.execute_cmd += ' ' + str(self.mpi_rank) + ' '
            self.exit_cmd = config['test_configuration']['protocol']['exit_cmd']
            self.exit_cmd += ' ' + str(self.mpi_rank) + ' '
            self.ip_sub_pub_list = { str(env_id): ip_sub_pub_first + str(self.mpi_rank) + str(env_id) for env_id in range(self.n_env)}
            self.ip_req_rep_list = { str(env_id): ip_req_rep_first + str(self.mpi_rank) + str(env_id) for env_id in range(self.n_env)}

    def run_experiment(self, env_idx, policy_param, valfn_param):
        assert( ( len(policy_param) - 1 ) / 2 == self.n_layer+1)
        assert( ( len(valfn_param) / 2 ) == self.n_layer+1)
        self.process_manager_list[str(env_idx)].execute_process()
        self.pair_and_sync(self.policy_valfn_socket_list[str(env_idx)],
                           self.data_socket_list[str(env_idx)])
        # ==================================================================
        # send policy
        # ==================================================================
        pb_policy_param = NeuralNetworkParam()
        for l_idx in range(self.n_layer + 1):
            weight = policy_param[2*l_idx]
            bias = policy_param[2*l_idx+1]
            layer = pb_policy_param.layers.add()
            layer.num_input = weight.shape[0]
            layer.num_output = weight.shape[1]
            for w_row in range(weight.shape[0]):
                for w_col in range(weight.shape[1]):
                    layer.weight.append(weight[w_row, w_col])
            for b_idx in range(bias.shape[0]):
                layer.bias.append(bias[b_idx])
            if l_idx == self.n_layer:
                layer.act_fn = NeuralNetworkParam.NONE
            else:
                layer.act_fn = self.act_fn
        for action_idx in range(policy_param[-1].shape[-1]):
            pb_policy_param.logstd.append((policy_param[-1])[0, action_idx])
        pb_policy_param_serialized = pb_policy_param.SerializeToString()
        self.policy_valfn_socket_list[str(env_idx)].send(pb_policy_param_serialized)
        self.policy_valfn_socket_list[str(env_idx)].recv()
        if self.verbose >= 1:
            print("[[Policy is set for %d th Env]]" % env_idx)

        # ==================================================================
        # send value function
        # ==================================================================
        pb_valfn_param = NeuralNetworkParam()
        for l_idx in range(self.n_layer + 1):
            weight = valfn_param[2*l_idx]
            bias = valfn_param[2*l_idx+1]
            layer = pb_valfn_param.layers.add()
            layer.num_input = weight.shape[0]
            layer.num_output = weight.shape[1]
            for w_row in range(weight.shape[0]):
                for w_col in range(weight.shape[1]):
                    layer.weight.append(weight[w_row, w_col])
            for b_idx in range(bias.shape[0]):
                layer.bias.append(bias[b_idx])
            if l_idx == self.n_layer:
                layer.act_fn = NeuralNetworkParam.NONE
            else:
                layer.act_fn = self.act_fn
        pb_valfn_param_serialized = pb_valfn_param.SerializeToString()
        self.policy_valfn_socket_list[str(env_idx)].send(pb_valfn_param_serialized)
        self.policy_valfn_socket_list[str(env_idx)].recv()
        if self.verbose >= 1:
            print("[[Value function is set for %d th Env]]" % env_idx)

    def pair_and_sync(self, req_rep_socket, sub_pub_socket):
        while True:
            try:
                zmq_msg = sub_pub_socket.recv(zmq.DONTWAIT)
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    req_rep_socket.send(b"nope")
                    req_rep_socket.recv()
                else:
                    raise
            else:
                req_rep_socket.send(b"world")
                req_rep_socket.recv()
                break;
        if self.verbose >= 1 :
            print("[[Sockets are all paired and synced]]")

    def run(self, policy_param, valfn_param):

        counts = np.zeros(shape=(self.n_steps, self.n_env), dtype=int)
        mb_obs = np.zeros(shape=(self.n_steps, self.n_env, self.ob_space.shape[0]), dtype=np.float32)
        mb_rewards = np.zeros(shape=(self.n_steps, self.n_env), dtype=np.float32)
        mb_actions = np.zeros(shape=(self.n_steps, self.n_env, self.ac_space.shape[0]), dtype=np.float32)
        actions_mean = np.zeros(shape=(self.n_steps, self.n_env, self.ac_space.shape[0]), dtype=np.float32)
        mb_values = np.zeros(shape=(self.n_steps, self.n_env), dtype=np.float32)
        mb_dones = np.zeros(shape=(self.n_steps, self.n_env), dtype=bool)
        mb_neglogpacs = np.zeros(shape=(self.n_steps, self.n_env), dtype=np.float32)
        mb_states = None
        last_values = np.zeros(shape=(self.n_env))
        epinfos=[]
        dataset_total_rew=0
        cur_ep_ret = np.zeros(shape=(self.n_env), dtype=np.float32)
        b_first = np.ones(shape=(self.n_env), dtype=bool)

        for env_idx in range(self.n_env):
            self.create_zmq_sockets(env_idx)
            self.run_experiment(env_idx, policy_param, valfn_param)

        for step_idx in tqdm(range(self.n_steps), ncols=80, desc="[Trajectory Roll Out]"):
        # for step_idx in range(self.n_steps):
            for env_idx in range(self.n_env):
                pb_data = Data()
                while(True):
                    zmq_msg = self.data_socket_list[str(env_idx)].recv()
                    if not (zmq_msg == b'hello'):
                        pb_data.ParseFromString(zmq_msg)
                        if pb_data.ListFields() == []:
                            assert(False)
                        else:
                            break
                counts[step_idx, env_idx] = pb_data.count
                if b_first[env_idx]:
                    assert(pb_data.count == 0)
                    b_first[env_idx] = False
                mb_obs[step_idx, env_idx] = pb_data.observation
                mb_rewards[step_idx, env_idx] = pb_data.reward
                mb_actions[step_idx, env_idx] = pb_data.action
                actions_mean[step_idx, env_idx] = pb_data.action_mean
                mb_values[step_idx, env_idx] = pb_data.value
                mb_dones[step_idx, env_idx] = pb_data.done
                mb_neglogpacs[step_idx, env_idx] = pb_data.neglogp
                cur_ep_ret[env_idx] += pb_data.reward
                dataset_total_rew += pb_data.reward
                if pb_data.done:
                    epinfos.append({'r':cur_ep_ret[env_idx],
                                    'l':pb_data.count})
                    cur_ep_ret[env_idx] = 0
                    self.process_manager_list[str(env_idx)].quit_process()
                    self.create_zmq_sockets(env_idx)
                    self.run_experiment(env_idx, policy_param, valfn_param)
                    b_first[env_idx] = True

        last_values = np.multiply(mb_values[-1], mb_dones[-1])

        for env_idx in range(self.n_env):
            self.process_manager_list[str(env_idx)].quit_process()

        # __import__('ipdb').set_trace()
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - mb_dones[-1]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_rewards, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, actions_mean)),
            mb_states, epinfos, dataset_total_rew)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
