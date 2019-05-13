import numpy as np
import tensorflow as tf
import gym
import roboschool
import time
import pickle
from spinup.algos.mwmsac import core
from spinup.algos.mwmsac.core import get_vars
from spinup.utils.logx import EpochLogger
import sys

"""
Meta World models Soft Actor-Critic ::: model based meta reinforcement learning
"""

# we do not need to update the hidden state of the target network in the get_action function
# since it's q value estimate is not used by the main network to compute mu and pi
def get_action(sess, mu, pi, x_ph, x_prev_ph, a_prev_ph, r_prev_ph, env_init,
               ctrl_init_1, ctrl_init_2, ctrl_init_3, env_hidden,
               ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3, o, o_prev,
               a_prev, r_prev, env_hidden_val, ctrl_hidden_1_val,
               ctrl_hidden_2_val, ctrl_hidden_3_val, deterministic=False):
    act_op = mu if deterministic else pi
    feed_dict = {x_ph: o.reshape(1, -1), x_prev_ph: o_prev.reshape(1, -1), a_prev_ph: a_prev.reshape(1, -1), r_prev_ph:np.array([r_prev])}
    if(env_hidden_val != None):
        feed_dict[env_init] = env_hidden_val
    if(ctrl_hidden_1_val != None):
        feed_dict[ctrl_init_1] = ctrl_hidden_1_val
    if(ctrl_hidden_2_val != None):
        feed_dict[ctrl_init_2] = ctrl_hidden_2_val
    if(ctrl_hidden_3_val != None):
        feed_dict[ctrl_init_3] = ctrl_hidden_3_val
    a, env_hidden_val, ctrl_hidden_1_val,\
    ctrl_hidden_2_val, ctrl_hidden_3_val = sess.run([act_op, env_hidden, ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3], feed_dict=feed_dict)
    a = a[0]
    return (a, env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val, ctrl_hidden_3_val)

def test_agent(sess, mu, pi, q1, q2, q1_pi, q2_pi, x_ph, x_prev_ph, a_prev_ph,
               r_prev_ph, env_init, ctrl_init_1, ctrl_init_2, ctrl_init_3,
               env_hidden, ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3,
               test_env, act_dim, max_ep_len, logger, n=10):
    # global sess, mu, pi, q1, q2, q1_pi, q2_pi
    env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val, ctrl_hidden_3_val = None, None, None, None
    for j in range(n):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        o_prev, a_prev, r_prev = o, np.zeros(act_dim), 0
        # Reset the hidden state of env model LSTM and controller LSTM here if you don't want it to be shared across episodes
        # env_hidden_val, ctrl_hidden_val = None, None
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a, env_hidden_val, ctrl_hidden_1_val, \
            ctrl_hidden_2_val, ctrl_hidden_3_val = get_action(sess, mu, pi, x_ph, x_prev_ph, a_prev_ph,
                                                              r_prev_ph, env_init, ctrl_init_1, ctrl_init_2,
                                                              ctrl_init_3, env_hidden, ctrl_hidden_1,
                                                              ctrl_hidden_2, ctrl_hidden_3, o, o_prev, a_prev,
                                                              r_prev, env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val,
                                                              ctrl_hidden_3_val, deterministic=True)
            o_prev, a_prev = o, a
            o, r, d, _ = test_env.step(a)
            r_prev = r
            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

def train(train_component, num_episode, sess, env, test_env, logger, save_freq, epochs, start_time,
          steps_per_epoch, steps_start, steps_end, max_ep_len, obs_dim, act_dim, x_ph, x_prev_ph, x2_ph,
          a_ph, a_prev_ph, r_ph, r_prev_ph, d_ph, env_init, ctrl_init_1, ctrl_init_2,
          ctrl_init_3, pi_loss, q1_loss, q2_loss, v_loss, auto_enc_loss, env_model_loss, mu, pi, q1,
          q2, q1_pi, q2_pi, v, logp_pi, env_hidden, ctrl_hidden_1, ctrl_hidden_2,
          ctrl_hidden_3, train_pi_op, train_value_op, train_auto_enc_op, train_env_model_op,
          target_update, o, r, d, ep_ret, ep_len, o_list, o_prev_list, o_prev,
          a_prev, r_prev, o2_list, r_list, r_prev_list, a_list, a_prev_list, d_list,
          env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val, ctrl_hidden_3_val,
          env_hidden_val_train, ctrl_hidden_1_val_train, ctrl_hidden_2_val_train, ctrl_hidden_3_val_train):

    num_train_repetition = 5
    if (train_component == 'world_model'):
        for eps in range(1, num_episode+1):
            fp = open(logger.output_dir+'/env_interation_data_'+str(eps)+'.pickle', 'rb')
            env_interation_data = pickle.load(fp)
            fp.close()
            o_list = env_interation_data['o_list']; o_prev_list = env_interation_data['o_prev_list']
            o2_list = env_interation_data['o2_list']; r_list = env_interation_data['r_list']
            r_prev_list = env_interation_data['r_prev_list']; a_list = env_interation_data['a_list']
            a_prev_list = env_interation_data['a_prev_list']; d_list = env_interation_data['d_list']

            for j in range(num_train_repetition):
                # Reset the hidden state values if you don't want it to be shared it across episodes during training,
                # we here do share it across the episodes during training as well in meta-RL
                # env_hidden_val_train, ctrl_hidden_val_train = None, None

                # for truncated back-prop through time of sequence length 40
                train_seq_len = 50
                num_seq = int(ep_len / train_seq_len)
                if (ep_len % train_seq_len != 0):
                    num_seq += 1
                env_model_loss_val_accum = 0.0
                # TODO:::: NOTE:::: Only if I throw away the data of previous episodes then we use ""start_idx = 0"" otherwise ""start_idx = -ep_len""
                start_idx = 0
                # start_idx = -ep_len
                end_idx = start_idx + train_seq_len
                for seq in range(num_seq):
                    if (end_idx >= 0):
                        end_idx = len(r_list)
                    feed_dict = {x_ph: np.vstack(o_list).reshape((-1, obs_dim))[start_idx:end_idx],
                                 x_prev_ph: np.vstack(o_prev_list).reshape((-1, obs_dim))[start_idx:end_idx],
                                 x2_ph: np.vstack(o2_list).reshape((-1, obs_dim))[start_idx:end_idx],
                                 a_ph: np.vstack(a_list).reshape((-1, act_dim))[start_idx:end_idx],
                                 a_prev_ph: np.vstack(a_prev_list).reshape((-1, act_dim))[start_idx:end_idx],
                                 r_ph: np.array(r_list)[start_idx:end_idx],
                                 r_prev_ph: np.array(r_prev_list)[start_idx:end_idx],
                                 d_ph: np.array(d_list)[start_idx:end_idx],
                                 }
                    if (env_hidden_val_train != None):
                        feed_dict[env_init] = env_hidden_val_train
                    if (ctrl_hidden_1_val_train != None):
                        feed_dict[ctrl_init_1] = ctrl_hidden_1_val_train
                    if (ctrl_hidden_2_val_train != None):
                        feed_dict[ctrl_init_2] = ctrl_hidden_2_val_train
                    if (ctrl_hidden_3_val_train != None):
                        feed_dict[ctrl_init_3] = ctrl_hidden_3_val_train

                    env_model_loss_val, env_hidden_val_train = sess.run([env_model_loss, env_hidden], feed_dict)
                    train_env_model_op_val, target_update_val = sess.run([train_env_model_op, target_update], feed_dict)
                    env_model_loss_val_accum += env_model_loss_val

                    start_idx += train_seq_len
                    end_idx += train_seq_len
                print(env_model_loss_val_accum)
        return num_episode
    # Main loop: collect experience in env and update/log each epoch
    for t in range(steps_start, steps_end):
        if(train_component=='auto_enc'):
            a = env.action_space.sample()
        elif(train_component=='ctrler_world_model'):
            a, env_hidden_val,  ctrl_hidden_1_val, \
            ctrl_hidden_2_val, ctrl_hidden_3_val = get_action(sess, mu, pi, x_ph, x_prev_ph, a_prev_ph,
                                                              r_prev_ph, env_init, ctrl_init_1, ctrl_init_2,
                                                              ctrl_init_3, env_hidden, ctrl_hidden_1,
                                                              ctrl_hidden_2, ctrl_hidden_3, o, o_prev, a_prev,
                                                              r_prev, env_hidden_val, ctrl_hidden_1_val,
                                                              ctrl_hidden_2_val, ctrl_hidden_3_val)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        o_list.append(np.array(o)); o_prev_list.append(np.array(o_prev)); o2_list.append(np.array(o2))
        r_list.append(r); r_prev_list.append(r_prev); a_list.append(np.array(a)); a_prev_list.append(np.array(a_prev)); d_list.append(d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o_prev, a_prev, r_prev = o, a, r
        o = o2

        if d or (ep_len == max_ep_len):
            num_episode+=1
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            # TODO change this back to range(ep_len) instead of range(50)
            if(train_component == 'auto_enc'):
                num_train_repetition = 5
            for j in range(num_train_repetition):
                # Reset the hidden state values if you don't want it to be shared it across episodes during training,
                # we here do share it across the episodes during training as well in meta-RL
                # env_hidden_val_train, ctrl_hidden_val_train = None, None

                # for truncated back-prop through time of sequence length 40
                train_seq_len = 50
                num_seq = int(ep_len/train_seq_len)
                if(ep_len%train_seq_len!=0):
                    num_seq+=1
                pi_loss_val_accum, q1_loss_val_accum, q2_loss_val_accum, v_loss_val_accum, auto_enc_loss_val_accum, env_model_loss_val_accum = [0.0]*6
                q1_val_accum, q2_val_accum, v_val_accum, logp_pi_val_accum = [np.array([0.0])] * 4
                # TODO:::: NOTE:::: Only if I throw away the data of previous episodes then we use ""start_idx = 0"" otherwise ""start_idx = -ep_len""
                start_idx = 0
                # start_idx = -ep_len
                end_idx = start_idx+train_seq_len
                for seq in range(num_seq):
                    if(end_idx>=0):
                        end_idx = len(r_list)
                    feed_dict = {x_ph: np.vstack(o_list).reshape((-1, obs_dim))[start_idx:end_idx],
                                 x_prev_ph: np.vstack(o_prev_list).reshape((-1, obs_dim))[start_idx:end_idx],
                                 x2_ph: np.vstack(o2_list).reshape((-1, obs_dim))[start_idx:end_idx],
                                 a_ph: np.vstack(a_list).reshape((-1, act_dim))[start_idx:end_idx],
                                 a_prev_ph: np.vstack(a_prev_list).reshape((-1, act_dim))[start_idx:end_idx],
                                 r_ph: np.array(r_list)[start_idx:end_idx],
                                 r_prev_ph: np.array(r_prev_list)[start_idx:end_idx],
                                 d_ph: np.array(d_list)[start_idx:end_idx],
                                 }
                    if(env_hidden_val_train != None):
                        feed_dict[env_init] = env_hidden_val_train
                    if(ctrl_hidden_1_val_train != None):
                        feed_dict[ctrl_init_1] = ctrl_hidden_1_val_train
                    if(ctrl_hidden_2_val_train != None):
                        feed_dict[ctrl_init_2] = ctrl_hidden_2_val_train
                    if(ctrl_hidden_3_val_train != None):
                        feed_dict[ctrl_init_3] = ctrl_hidden_3_val_train

                    #TODO ******* IMPORTANT ******** First train the auto-encoder alone and then trian the env model and then finally train the controller
                    if(train_component == 'auto_enc'):
                        auto_enc_loss_val, env_model_loss_val, env_hidden_val_train = sess.run([auto_enc_loss, env_model_loss, env_hidden], feed_dict)
                        train_auto_enc_op_val, target_update_val = sess.run([train_auto_enc_op, target_update], feed_dict)
                        auto_enc_loss_val_accum += auto_enc_loss_val
                    elif(train_component == 'ctrler_world_model'):
                        pi_loss_val, q1_loss_val, q2_loss_val, v_loss_val, \
                        auto_enc_loss_val, env_model_loss_val, q1_val, q2_val,\
                        v_val, logp_pi_val, env_hidden_val_train, ctrl_hidden_1_val_train,\
                        ctrl_hidden_2_val_train, ctrl_hidden_3_val_train = sess.run([pi_loss, q1_loss,
                                                                                     q2_loss, v_loss,
                                                                                     auto_enc_loss, env_model_loss, q1, q2,
                                                                                     v, logp_pi, env_hidden,
                                                                                     ctrl_hidden_1, ctrl_hidden_2,
                                                                                     ctrl_hidden_3], feed_dict)
                        train_auto_enc_op_val, train_env_model_op_val, \
                        train_pi_op_val, train_value_op_val, target_update_val = sess.run([train_auto_enc_op, train_env_model_op,
                                                                                           train_pi_op, train_value_op,
                                                                                           target_update], feed_dict)

                        pi_loss_val_accum += pi_loss_val; q1_loss_val_accum += q1_loss_val
                        q2_loss_val_accum += q2_loss_val; v_loss_val_accum += v_loss_val
                        auto_enc_loss_val_accum += auto_enc_loss_val
                        env_model_loss_val_accum += env_model_loss_val
                        q1_val_accum = np.concatenate((q1_val_accum, q1_val))
                        q2_val_accum = np.concatenate((q2_val_accum, q2_val))
                        v_val_accum = np.concatenate((v_val_accum, v_val))
                        logp_pi_val_accum = np.concatenate((logp_pi_val_accum, logp_pi_val))

                    start_idx+=train_seq_len
                    end_idx+=train_seq_len

                if(train_component == 'ctrler_world_model'):
                    q1_val_accum = q1_val_accum[1:]; q2_val_accum = q2_val_accum[1:]
                    v_val_accum = v_val_accum[1:]; logp_pi_val_accum = logp_pi_val_accum[1:]
                logger.store(LossPi=pi_loss_val_accum, LossQ1=q1_loss_val_accum,
                             LossQ2=q2_loss_val_accum, LossV=v_loss_val_accum,
                             AutoEncLoss=auto_enc_loss_val_accum, EnvModelLoss=env_model_loss_val_accum,
                             Q1Vals=q1_val_accum, Q2Vals=q2_val_accum, VVals=v_val_accum, LogPi=logp_pi_val_accum)

                # feed_dict = {x_ph: np.vstack(o_list).reshape((-1, obs_dim))[-ep_len:],
                #             x_prev_ph: np.vstack(o_prev_list).reshape((-1, obs_dim))[-ep_len:],
                #             x2_ph: np.vstack(o2_list).reshape((-1, obs_dim))[-ep_len:],
                #             a_ph: np.vstack(a_list).reshape((-1, act_dim))[-ep_len:],
                #             a_prev_ph: np.vstack(a_prev_list).reshape((-1, act_dim))[-ep_len:],
                #             r_ph: np.array(r_list)[-ep_len:],
                #             r_prev_ph: np.array(r_prev_list)[-ep_len:],
                #             d_ph: np.array(d_list)[-ep_len:],
                #             }
                # # TODO : share the hidden states across episodes here
                # if (env_hidden_val != None):
                #     feed_dict[env_init] = env_hidden_val_train
                # if (ctrl_hidden_val != None):
                #     feed_dict[ctrl_init] = ctrl_hidden_val_train
                #
                # pi_loss_val, q1_loss_val, q2_loss_val, v_loss_val, \
                # env_model_loss_val, q1_val, q2_val, v_val, \
                # logp_pi_val, env_hidden_val_train, ctrl_hidden_val_train = sess.run([pi_loss, q1_loss, q2_loss, v_loss,
                #                                                                        env_model_loss, q1, q2, v, logp_pi,
                #                                                                        env_init, ctrl_hidden], feed_dict)
                # train_pi_op_val, train_value_op_val, \
                # train_env_model_op_val, target_update_val = sess.run([train_pi_op, train_value_op,
                #                                                       train_env_model_op, target_update], feed_dict)
                #
                # logger.store(LossPi=pi_loss_val, LossQ1=q1_loss_val, LossQ2=q2_loss_val,
                #          LossV=v_loss_val, EnvModelLoss=env_model_loss_val, Q1Vals=q1_val, Q2Vals=q2_val,
                #          VVals=v_val, LogPi=logp_pi_val)
                # # logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                # #              LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                # #              VVals=outs[6], LogPi=outs[7])


            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            # TODO:::: NOTE:::: I throw away the data of previous episodes,
            # TODO:::: as it unnecessarily increases the size of the total data stored, which I believe is slowing the code down considerably
            if (train_component == 'auto_enc'):
                env_interation_data = {'o_list':o_list, 'o_prev_list':o_prev_list,
                                       'o2_list':o2_list, 'r_list':r_list,
                                       'r_prev_list':r_prev_list, 'a_list':a_list,
                                       'a_prev_list':a_prev_list, 'd_list':d_list}
                fp = open(logger.output_dir+'/env_interation_data_'+str(num_episode)+'.pickle', 'wb')
                pickle.dump(env_interation_data, fp)
                fp.close()
            o_list, o_prev_list, o2_list, r_list, r_prev_list, a_list, a_prev_list, d_list = [], [], [], [], [], [], [], []
            # Reset the hidden state of env model LSTM and controller LSTM here if you don't want it to be shared across episodes
            # env_hidden_val, ctrl_hidden_val = None, None

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(sess, mu, pi, q1, q2, q1_pi, q2_pi, x_ph, x_prev_ph,
                       a_prev_ph, r_prev_ph, env_init, ctrl_init_1, ctrl_init_2,
                       ctrl_init_3, env_hidden, ctrl_hidden_1, ctrl_hidden_2,
                       ctrl_hidden_3, test_env, act_dim, max_ep_len, logger)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('AutoEncLoss', average_only=True)
            logger.log_tabular('EnvModelLoss', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    return num_episode

def mwmsac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=15*5000,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    np.random.seed(seed)
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    with tf.device('/gpu:0'):
    # with tf.device(tf.test.gpu_device_name()):
        tf.set_random_seed(seed)
        # Inputs to computation graph
        x_ph, x_prev_ph, a_ph, a_prev_ph, x2_ph, r_ph, r_prev_ph, d_ph = core.placeholders(obs_dim, obs_dim, act_dim, act_dim, obs_dim, None, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            ops_dict = actor_critic(x_ph, x_prev_ph, a_ph, a_prev_ph, r_prev_ph, **ac_kwargs)
        mu = ops_dict['mu']; pi = ops_dict['pi']; logp_pi=ops_dict['logp_pi']; q1 = ops_dict['q1']; q2 = ops_dict['q2']
        q1_pi = ops_dict['q1_pi']; q2_pi = ops_dict['q2_pi']; v = ops_dict['v']
        x_latent = ops_dict['x_latent']; latent_vec_dim = ops_dict['latent_vec_dim']
        x_decoded = ops_dict['x_decoded']; x_prev_decoded = ops_dict['x_prev_decoded']
        x_vae_mu = ops_dict['x_vae_mu']; x_vae_logvar = ops_dict['x_vae_logvar']
        x_prev_vae_mu = ops_dict['x_prev_vae_mu']; x_prev_vae_logvar = ops_dict['x_prev_vae_logvar']
        # env_model_predicted_state = ops_dict['env_model_predicted_state']
        mu_sigma_pi = ops_dict['mu_sigma_pi']
        env_hidden = ops_dict['env_hidden']; env_init = ops_dict['env_init']
        ctrl_hidden_1 = ops_dict['ctrl_hidden_1']; ctrl_init_1 = ops_dict['ctrl_init_1']
        ctrl_hidden_2 = ops_dict['ctrl_hidden_2']; ctrl_init_2 = ops_dict['ctrl_init_2']
        ctrl_hidden_3 = ops_dict['ctrl_hidden_3']; ctrl_init_3 = ops_dict['ctrl_init_3']
        # x_encoded = ops_dict['x_encoded']

        # Target value network
        with tf.variable_scope('target'):
            # note: passing a_ph to the target network doesn't really matter, since a_ph will not be used
            # in neither the calculation of v_targ nor the calculation of env_hidden_targ
            # note: mu however will be used by the target network to calculate v_targ
            ops_dict_targ = actor_critic(x2_ph, x_ph, a_ph, mu, r_ph, **ac_kwargs)
            v_targ = ops_dict_targ['v']

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in
                           ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main/env_model', 'main/state_encoder', 'main/controller', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t' + \
               'q1: %d, \t q2: %d, \t v: %d, \t env_model: %d, \t state_encoder: %d, \t controller: %d, \t total: %d\n') % var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * v_targ)
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(alpha*logp_pi - q1_pi)
        q1_loss = 0.5*tf.reduce_mean((q_backup - q1)**2)
        q2_loss = 0.5*tf.reduce_mean((q_backup - q2)**2)
        v_loss = 0.5*tf.reduce_mean((v_backup - v)**2)
        value_loss = q1_loss + q2_loss + v_loss
        # # The env model is made to predict the next state instead of abstract next state, as it removes the non-stationarity in its targets
        # The env model is made to predict the abstract next state, to remove the non-stationarity in its targets, we pre-train the VAE module
        # MDN reference citation : https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py
        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        def tf_lognormal(y, mean, logstd):
            return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI
        def get_mdn_coef(output):
            logmix, mean, logstd = tf.split(output, 3, 1)
            logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
            return logmix, mean, logstd
        def get_lossfunc(logmix, mean, logstd, y):
            v = logmix + tf_lognormal(y, mean, logstd)
            v = tf.reduce_logsumexp(v, 1, keepdims=True)
            return -tf.reduce_mean(v)
        out_logmix, out_mean, out_logstd = get_mdn_coef(mu_sigma_pi)
        # reshape target data so that it is compatible with prediction shape
        #TODO ***** IMPORTANT ****** :::: Check that the target output here is correct
        flat_target_data = tf.reshape(x_latent, [-1, 1])
        lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)
        env_model_loss = tf.reduce_mean(lossfunc)
        # env_model_loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(env_model_predicted_state - x_latent), axis=1), axis=0)
        # env_model_loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(env_model_predicted_state - x_ph), axis=1), axis=0)
        # env_model_loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(env_model_predicted_state - x_encoded), axis=1), axis=0)
        # VAE reconstruction loss
        recons_loss_x = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(tf.sigmoid(x_ph) - x_decoded), axis=1), axis=0)
        recons_loss_x_prev = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(tf.sigmoid(x_prev_ph) - x_prev_decoded), axis=1), axis=0)
        # augmented kl loss per dim
        kl_tolerance = 0.5
        kl_loss_x = - 0.5*tf.reduce_sum((1.0 + x_vae_logvar - tf.square(x_vae_mu) - tf.exp(x_vae_logvar)), reduction_indices=1)
        # kl_loss_x = tf.maximum(kl_loss_x, kl_tolerance*latent_vec_dim)
        kl_loss_x = tf.reduce_mean(kl_loss_x)
        kl_loss_x_prev = - 0.5 * tf.reduce_sum((1.0 + x_prev_vae_logvar - tf.square(x_prev_vae_mu) - tf.exp(x_prev_vae_logvar)), reduction_indices=1)
        # kl_loss_x_prev = tf.maximum(kl_loss_x_prev, kl_tolerance*latent_vec_dim)
        kl_loss_x_prev = tf.reduce_mean(kl_loss_x_prev)
        auto_enc_loss = 0.5*recons_loss_x + 0.5*recons_loss_x_prev + 0.5*kl_loss_x + 0.5*kl_loss_x_prev

        # Auto encoder train op
        auto_enc_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # # TODO ******IMPORTANT******* :::: Use gradient clipping on env model loss
        # auto_enc_optimizer = tf.contrib.estimator.clip_gradients_by_norm(auto_enc_optimizer, clip_norm=10.0)
        auto_enc_params = get_vars('main/state_encoder') + get_vars('main/VAE') + get_vars('main/state_decoder')
        train_auto_enc_op = auto_enc_optimizer.minimize(auto_enc_loss, var_list=auto_enc_params)

        # Env model train op
        env_model_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # # TODO:::: Use gradient clipping on env model loss
        # env_model_optimizer = tf.contrib.estimator.clip_gradients_by_norm(env_model_optimizer, clip_norm=10.0)
        # NOTE:::: The env model loss is not backpropgated through the state_enocder otherwise
        #  it will be chasing its own tail, in terms of non-stationarity of targets
        # and may result in degenerate solutions
        env_model_params = get_vars('main/env_model')
        # env_model_params = get_vars('main/env_model') + get_vars('main/state_encoder')
        with tf.control_dependencies([train_auto_enc_op]):
            train_env_model_op = env_model_optimizer.minimize(env_model_loss, var_list=env_model_params)

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # # TODO:::: Use gradient clipping on pi loss, since its values are very high
        # pi_optimizer = tf.contrib.estimator.clip_gradients_by_norm(pi_optimizer, clip_norm=10.0)
        # TODO *****IMPORTANT******:::: Figure out whether the policy loss should be backprogpated through the controller RNN or not, i believe it should be
        # pi_params = get_vars('main/pi')
        pi_params = get_vars('main/pi') + get_vars('main/controller')
        # pi_params = get_vars('main/pi') + get_vars('main/state_encoder') + get_vars('main/controller')

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # # TODO:::: Use gradient clipping on value loss
        # value_optimizer = tf.contrib.estimator.clip_gradients_by_norm(value_optimizer, clip_norm=10.0)
        # TODO:::: NOTE:::::: Even the Q/V loss should not be back-propagated through the state_encoder
        # TODO:::: features, the state_encoder weights should be trained only with reconstruction loss (VAE) style
        # value_params = get_vars('main/q') + get_vars('main/v') + get_vars('main/state_encoder') + get_vars('main/controller')
        value_params = get_vars('main/q') + get_vars('main/v') + get_vars('main/controller')

        with tf.control_dependencies([train_env_model_op]):
            train_pi_op = pi_optimizer.minimize(pi_loss, var_list=pi_params)
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_pi_op, train_value_op, train_env_model_op]):
            target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        # step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
        #             train_pi_op,train_value_op, train_env_model_op, target_update]

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # sess = tf.Session(config=config)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'x_prev':x_prev_ph, 'a': a_ph, 'a_prev':a_prev_ph},
                          outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o_prev, a_prev, r_prev = o, np.zeros(act_dim), 0
    total_steps = steps_per_epoch * epochs

    # TODO MUST TODO::::: Use np.array as the data structure to store the experience, I guess using lists is considerably slower in comparison
    o_list, o_prev_list, o2_list, r_list, r_prev_list, a_list, a_prev_list, d_list = [], [], [], [], [], [], [], []
    env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val, ctrl_hidden_3_val = None, None, None, None
    env_hidden_val_train, ctrl_hidden_1_val_train, ctrl_hidden_2_val_train, ctrl_hidden_3_val_train = None, None, None, None

    # Pre-training Variational Auto encoder(V) loop: collect experience to learn the variational auto-encoder which we will be later used by the controller.
    num_episode = 0
    num_episode = train('auto_enc', num_episode, sess, env, test_env, logger, save_freq, epochs, start_time, steps_per_epoch,
                          0, start_steps, max_ep_len, obs_dim, act_dim, x_ph, x_prev_ph, x2_ph, a_ph, a_prev_ph,
                          r_ph, r_prev_ph, d_ph, env_init, ctrl_init_1, ctrl_init_2, ctrl_init_3, pi_loss,
                          q1_loss, q2_loss, v_loss, auto_enc_loss, env_model_loss, mu, pi, q1, q2, q1_pi, q2_pi, v, logp_pi,
                          env_hidden, ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3, train_pi_op, train_value_op,
                          train_auto_enc_op, train_env_model_op, target_update, o, r, d, ep_ret, ep_len, o_list, o_prev_list,
                          o_prev, a_prev, r_prev, o2_list, r_list, r_prev_list, a_list, a_prev_list, d_list,
                          env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val, ctrl_hidden_3_val,
                          env_hidden_val_train, ctrl_hidden_1_val_train, ctrl_hidden_2_val_train, ctrl_hidden_3_val_train)

    # Pre-training World model loop: collect experience to learn the world models which we will be later used by the controller.
    num_episode = train('world_model', num_episode, sess, env, test_env, logger, save_freq, epochs, start_time, steps_per_epoch,
                          0, start_steps, max_ep_len, obs_dim, act_dim, x_ph, x_prev_ph, x2_ph, a_ph, a_prev_ph,
                          r_ph, r_prev_ph, d_ph, env_init, ctrl_init_1, ctrl_init_2, ctrl_init_3, pi_loss,
                          q1_loss, q2_loss, v_loss, auto_enc_loss, env_model_loss, mu, pi, q1, q2, q1_pi, q2_pi, v, logp_pi,
                          env_hidden, ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3, train_pi_op, train_value_op,
                          train_auto_enc_op, train_env_model_op, target_update, o, r, d, ep_ret, ep_len, o_list, o_prev_list,
                          o_prev, a_prev, r_prev, o2_list, r_list, r_prev_list, a_list, a_prev_list, d_list,
                          env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val, ctrl_hidden_3_val,
                          env_hidden_val_train, ctrl_hidden_1_val_train, ctrl_hidden_2_val_train, ctrl_hidden_3_val_train)

    num_episode = train('ctrler_world_model', num_episode, sess, env, test_env, logger, save_freq, epochs, start_time, steps_per_epoch,
                          start_steps, total_steps, max_ep_len, obs_dim, act_dim, x_ph, x_prev_ph, x2_ph, a_ph, a_prev_ph,
                          r_ph, r_prev_ph, d_ph, env_init, ctrl_init_1, ctrl_init_2, ctrl_init_3, pi_loss,
                          q1_loss, q2_loss, v_loss, auto_enc_loss, env_model_loss, mu, pi, q1, q2, q1_pi, q2_pi, v, logp_pi,
                          env_hidden, ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3, train_pi_op, train_value_op,
                          train_auto_enc_op, train_env_model_op, target_update, o, r, d, ep_ret, ep_len, o_list, o_prev_list,
                          o_prev, a_prev, r_prev, o2_list, r_list, r_prev_list, a_list, a_prev_list, d_list,
                          env_hidden_val, ctrl_hidden_1_val, ctrl_hidden_2_val, ctrl_hidden_3_val,
                          env_hidden_val_train, ctrl_hidden_1_val_train, ctrl_hidden_2_val_train, ctrl_hidden_3_val_train)

    sess.close()

# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--env', type=str, default='Ant-v2')
#     parser.add_argument('--env', type=str, default='Pendulum-v0')
#     parser.add_argument('--hid', type=int, default=300)
#     parser.add_argument('--l', type=int, default=1)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='mwmsac')
#     args = parser.parse_args()
#
#     from spinup.utils.run_utils import setup_logger_kwargs
#
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
#
#     mwmsac(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
#         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
#         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#         logger_kwargs=logger_kwargs)

if __name__ == '__main__':
    from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs('mwmsac_roboscl_Inv_Pendulum_V1_5_iter_PRE_train_No_enc', 0)
    # logger_kwargs = setup_logger_kwargs('mwmsac_roboscl_Ant_V1_5_iter_resnet_PRE_VAE_train_all_unclipped_VAE_MDN_Stacked_LSTM', 0)
    logger_kwargs = setup_logger_kwargs('mwmsac_roboscl_Ant_V1_5_iter_resnet_PRE_sigmoid_VAE_train_all_unclipped_VAE_MDN_Stacked_LSTM', 0)
    # logger_kwargs = setup_logger_kwargs('mwmsac_roboscl_Half_Cheetah_5_iter_resnet_PRE_VAE_train_all_unclipped_VAE_MDN_Stacked_LSTM', 0)
    # logger_kwargs = setup_logger_kwargs('mwmsac_roboscl_Ant_5_iter_resnet_PRE_VAE_train_unclipped_VAE_MDN_Stacked_LSTM', 0)
    mwmsac(lambda: gym.make('RoboschoolAnt-v1'), actor_critic=core.mlp_actor_critic,
    # mwmsac(lambda: gym.make('RoboschoolHalfCheetah-v1'), actor_critic=core.mlp_actor_critic,
    # mwmsac(lambda: gym.make('RoboschoolInvertedPendulum-v1'), actor_critic=core.mlp_actor_critic,
        logger_kwargs=logger_kwargs)