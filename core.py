import numpy as np
import tensorflow as tf

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, act_dim, hidden_sizes, activation, output_activation):
    act_dim = act_dim
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi

# Wrapper reference citation : https://stackoverflow.com/questions/47745027/tensorflow-how-to-obtain-intermediate-cell-states-c-from-lstmcell-using-dynam
class Wrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inner_cell):
        super(Wrapper, self).__init__()
        self._inner_cell = inner_cell
    @property
    def state_size(self):
        return self._inner_cell.state_size
    @property
    def output_size(self):
        return (self._inner_cell.state_size, self._inner_cell.output_size)
    def call(self, input, *args, **kwargs):
        output, next_state = self._inner_cell(input, *args, **kwargs)
        emit_output = (next_state, output)
        return emit_output, next_state

def single_cell(shape):
    # CuDNN LSTM doesn't support variable length input yet, so can't use it!!!
    # return Wrapper(tf.keras.layers.CuDNNLSTM(num_layers=1, num_units=shape, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='env_model_lstm_cell'))
    # return Wrapper(tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=shape, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='env_model_lstm_cell'))
    # TODO:::: try using GRU as well
    # return Wrapper(tf.nn.rnn_cell.LSTMCell(shape, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='env_model_lstm_cell'))
    return Wrapper(tf.nn.rnn_cell.LSTMCell(shape, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='env_model_lstm_cell'))

def state_encoder(state, reuse_weights=False):
    state_encoder_neurons = 30
    latent_vec_dim = 15
    state_dim = int(state.shape[1])
    # Encoder
    with tf.variable_scope('state_encoder', reuse=reuse_weights):
        state = tf.sigmoid(state)
        h_enc_1 = tf.contrib.layers.fully_connected(state, state_encoder_neurons, activation_fn=tf.nn.leaky_relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1), trainable=True, scope='enc_1')
        h_enc_2 = tf.contrib.layers.fully_connected(h_enc_1, state_encoder_neurons, activation_fn=tf.nn.leaky_relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1), trainable=True, scope='enc_2')
        # h_enc_1 = tf.layers.dense(state, units=state_encoder_neurons, activation=tf.nn.leaky_relu, reuse=reuse_weights)
        # h_enc_2 = tf.layers.dense(h_enc_1, units=state_encoder_neurons, activation=tf.nn.leaky_relu, reuse=reuse_weights)
        encoded_state = h_enc_2

    # VAE
    # VAE reference citation : https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
    with tf.variable_scope('VAE', reuse=reuse_weights):
        vae_mu = tf.contrib.layers.fully_connected(encoded_state, latent_vec_dim,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.constant_initializer(0.1), trainable=True, scope='vae_mu')
        # vae_mu = tf.layers.dense(encoded_state, latent_vec_dim, reuse=reuse_weights, name="vae_mu")
        vae_logvar = tf.contrib.layers.fully_connected(encoded_state, latent_vec_dim,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.constant_initializer(0.1), trainable=True, scope='log_var')
        # vae_logvar = tf.layers.dense(encoded_state, latent_vec_dim, reuse=reuse_weights, name="log_var")
        sigma = tf.exp(vae_logvar/2.0)
        epsilon = tf.random_normal([tf.shape(state)[0], latent_vec_dim])
        latent_vec = vae_mu + sigma*epsilon

    # Decoder
    with tf.variable_scope('state_decoder', reuse=reuse_weights):
        dec_fc = tf.contrib.layers.fully_connected(latent_vec, state_encoder_neurons,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1), trainable=True, scope='dec_0')
        h_dec_2 = tf.contrib.layers.fully_connected(dec_fc, state_encoder_neurons, activation_fn=tf.nn.leaky_relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1), trainable=True, scope='dec_1')
        h_dec_1 = tf.contrib.layers.fully_connected(h_dec_2, state_dim, activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1), trainable=True, scope='dec_2')
        # dec_fc = tf.layers.dense(latent_vec,  units=state_encoder_neurons, reuse=reuse_weights)
        # h_dec_2 = tf.layers.dense(dec_fc,  units=state_encoder_neurons, activation=tf.nn.leaky_relu, reuse=reuse_weights)
        # h_dec_1 = tf.layers.dense(h_dec_2, units=state_dim, activation=None, reuse=reuse_weights)
        decoded_state = h_dec_1
        decoded_state = tf.sigmoid(decoded_state)

    return latent_vec, encoded_state, decoded_state, vae_mu, vae_logvar, latent_vec_dim

    # # Identity encoder, either it is an identity encoder or it is a VAE trained on reconstruction loss, the encoder will not be trained on the Q/V loss.
    # return state, state, state, state, state, state.shape[1]

# MDN reference citation : https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py
def env_model(x_prev_encoded, a_prev, state_dim):
    num_mixtures = 5
    latent_vec_dim = 15
    nout = latent_vec_dim*num_mixtures*3
    # TODO ::: Use stacked LSTM for the env model and feed the state encoding forward to the subsequent layers of the stacked LSTM
    # TODO :::: ***** IMPORTANT ******* make the env model predict reward
    # state_encoder_neurons = 200
    state_encoder_neurons = state_dim
    env_model_lstm_neurons = 30
    env_model_layers = 1
    with tf.variable_scope('env_model'):
        x_prev_encoded = tf.reshape(x_prev_encoded, [1, -1, x_prev_encoded.shape[1]])
        a_prev = tf.reshape(a_prev, [1, -1, a_prev.shape[1]])
        env_model_input = tf.concat([x_prev_encoded, a_prev], axis=2)
        lstm_layers = tf.nn.rnn_cell.MultiRNNCell([single_cell(env_model_lstm_neurons) for _ in range(env_model_layers)])
        batch_size = int(env_model_input.shape[0])
        env_init = lstm_layers.zero_state(batch_size=batch_size, dtype=tf.float32)
        # dummy_lstm_cell = tf.nn.rnn_cell.LSTMCell(state_encoder_neurons, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='dummy_lstm_cell')
        # dummy_lstm_layers = tf.nn.rnn_cell.MultiRNNCell([dummy_lstm_cell for _ in range(env_model_layers)])
        # env_init = dummy_lstm_layers.zero_state(batch_size=batch_size, dtype=tf.float32)
        # env_init = (tf.placeholder(env_model_layers, batch_size, state_encoder_neurons), tf.placeholder(env_model_layers, batch_size, state_encoder_neurons))
        lstm_out, env_model_rnn_hidden_state = tf.nn.dynamic_rnn(cell=lstm_layers, inputs=env_model_input, initial_state=env_init)
        lstm_output = lstm_out[1]
        lstm_output_flat = tf.reshape(lstm_output, [-1, env_model_lstm_neurons])
        # TODO :::: ***** IMPORTANT ***** Try using the hidden state of the world model RNN, instead of its cell state
        env_model_cell_state = lstm_out[0].c
        env_model_cell_state = tf.reshape(env_model_cell_state, [-1, env_model_lstm_neurons])
        mu_sigma_pi = tf.layers.dense(lstm_output_flat, units=nout, activation=None)
        # NOTE:::: mu_sigma_pi is flattened such that it has dimension (batch_size*num_features) x (num_mixtures*3)
        mu_sigma_pi = tf.reshape(mu_sigma_pi, [-1, num_mixtures*3])
        # # The env model is made to predict the next state instead of abstract next state, as it removes the non-stationarity in its targets
        # next_encoded_state_1 = tf.layers.dense(lstm_output, units=state_encoder_neurons, activation=None)
        # # next_encoded_state_1 = tf.layers.dense(lstm_output, units=state_encoder_neurons, activation=tf.nn.leaky_relu)
        # next_encoded_state = tf.reshape(next_encoded_state_1, [-1, state_encoder_neurons])
        # # next_encoded_state_1 = tf.layers.dense(lstm_output, units=state_encoder_neurons, activation=None)
    return mu_sigma_pi, env_model_cell_state, env_model_rnn_hidden_state, env_init

def controller_rnn(x_latent, h, a_prev, r_prev):
    ctrl_rnn_neurons = 30
    ctrl_rnn_layers = 1
    # Stacked LSTM
    with tf.variable_scope('controller'):
        x_h_concatenated = tf.concat([x_latent, h], axis=1)
        x_h_concatenated = tf.reshape(x_h_concatenated, [1, -1, x_h_concatenated.shape[1]])
        # cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=controller_rnn_neurons, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='controller_lstm_cell')
        # cell_1 = tf.nn.rnn_cell.LSTMCell(ctrl_rnn_neurons, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='ctrl_cell_1')
        cell_1 = tf.nn.rnn_cell.LSTMCell(ctrl_rnn_neurons, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='ctrl_cell_1')
        # TODO:::: try using GRU as well
        lstm_layer_1 = tf.nn.rnn_cell.MultiRNNCell([cell_1])
        batch_size = int(x_h_concatenated.shape[0])
        # cell = tf.keras.layers.CuDNNLSTM(num_layers=1, num_units=controller_rnn_neurons, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='controller_lstm_cell')
        # dummy_lstm_cell = tf.nn.rnn_cell.LSTMCell(controller_rnn_neurons, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='dummy_cell')
        # dummy_lstm_layers = tf.nn.rnn_cell.MultiRNNCell([dummy_lstm_cell for _ in range(controller_rnn_layers)])
        # ctrler_init = dummy_lstm_layers.zero_state(batch_size=batch_size, dtype=tf.float32)
        # ctrler_init = (tf.placeholder(controller_rnn_layers, batch_size, controller_rnn_neurons), tf.placeholder(controller_rnn_layers, batch_size, controller_rnn_neurons))
        ctrl_init_1 = lstm_layer_1.zero_state(batch_size=batch_size, dtype=tf.float32)
        lstm_1, ctrl_hidden_1 = tf.nn.dynamic_rnn(cell=lstm_layer_1, inputs=x_h_concatenated, initial_state=ctrl_init_1)
        # NOTE ::: make the controller feed the state encoding forward to the subsequent layers of the stacked LSTM
        x_h_r_concatenated = tf.concat([x_latent, tf.reshape(lstm_1, [-1, ctrl_rnn_neurons]), r_prev], axis=1)
        x_h_r_concatenated = tf.reshape(x_h_r_concatenated, [1, -1, x_h_r_concatenated.shape[1]])
        # cell_2 = tf.nn.rnn_cell.LSTMCell(ctrl_rnn_neurons, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='ctrl_cell_2')
        cell_2 = tf.nn.rnn_cell.LSTMCell(ctrl_rnn_neurons, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='ctrl_cell_2')
        lstm_layer_2 = tf.nn.rnn_cell.MultiRNNCell([cell_2])
        ctrl_init_2 = lstm_layer_2.zero_state(batch_size=batch_size, dtype=tf.float32)
        lstm_2, ctrl_hidden_2 = tf.nn.dynamic_rnn(cell=lstm_layer_2, inputs=x_h_r_concatenated, initial_state=ctrl_init_2)
        # NOTE ::: make the controller feed the state encoding forward to the subsequent layers of the stacked LSTM
        x_h_r_a_concatenated = tf.concat([x_latent, tf.reshape(lstm_2, [-1, ctrl_rnn_neurons]), a_prev], axis=1)
        x_h_r_a_concatenated = tf.reshape(x_h_r_a_concatenated, [1, -1, x_h_r_a_concatenated.shape[1]])
        # cell_3 = tf.nn.rnn_cell.LSTMCell(ctrl_rnn_neurons, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='ctrl_cell_3')
        cell_3 = tf.nn.rnn_cell.LSTMCell(ctrl_rnn_neurons, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(), name='ctrl_cell_3')
        lstm_layer_3 = tf.nn.rnn_cell.MultiRNNCell([cell_3])
        ctrl_init_3 = lstm_layer_3.zero_state(batch_size=batch_size, dtype=tf.float32)
        lstm_3, ctrl_hidden_3 = tf.nn.dynamic_rnn(cell=lstm_layer_3, inputs=x_h_r_a_concatenated, initial_state=ctrl_init_3)
        hidden_1 = tf.reshape(lstm_3, [-1, ctrl_rnn_neurons])
    return hidden_1, ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3, ctrl_init_1, ctrl_init_2, ctrl_init_3

"""
Actor-Critics
"""
def mlp_actor_critic(x, x_prev, a, a_prev, r_prev, hidden_sizes=(100,100), activation=tf.nn.relu,
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # encode prev state
    x_prev_latent, x_prev_encoded, x_prev_decoded, x_prev_vae_mu, x_prev_vae_logvar, latent_vec_dim = state_encoder(x_prev, reuse_weights=False)
    # update the cell state of the world model
    state_dim = x.shape[1]
    mu_sigma_pi, env_model_cell_state, env_hidden, env_init = env_model(x_prev_latent, a_prev, state_dim)
    # env_model_predicted_state, env_model_cell_state, env_hidden, env_init = env_model(x_prev_latent, a_prev, state_dim)
    # encode current state
    x_latent, x_encoded, x_decoded, x_vae_mu, x_vae_logvar, latent_vec_dim = state_encoder(x, reuse_weights=True)
    # prepare input for the controller
    h = tf.stop_gradient(env_model_cell_state)
    # x_h_concatenated = tf.concat([x_latent, tf.stop_gradient(env_model_cell_state)], axis=1)
    controller_x, ctrl_hidden_1, ctrl_hidden_2, ctrl_hidden_3, ctrl_init_1, ctrl_init_2, ctrl_init_3 = controller_rnn(x_latent, h, a_prev, tf.reshape(r_prev, [-1,1]))
    controller_x = tf.concat([x_latent, h, controller_x], axis=1)
    act_dim = action_space.shape[0]
    # policy
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = policy(controller_x, act_dim, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    # vfs
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([controller_x,a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([controller_x,pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([controller_x,a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([controller_x,pi], axis=-1))
    with tf.variable_scope('v'):
        v = vf_mlp(controller_x)
    return {'mu':mu, 'pi':pi, 'logp_pi':logp_pi, 'q1':q1, 'q2':q2, 'q1_pi':q1_pi, 'q2_pi':q2_pi,
            'v':v, 'x_latent':x_latent,
            'x_decoded':x_decoded, 'x_prev_decoded': x_prev_decoded,
            'x_vae_mu':x_vae_mu, 'x_vae_logvar':x_vae_logvar,
            'x_prev_vae_mu':x_prev_vae_mu, 'x_prev_vae_logvar':x_prev_vae_logvar,
            'latent_vec_dim': latent_vec_dim,
            'env_hidden':env_hidden, 'env_init':env_init,
            'mu_sigma_pi': mu_sigma_pi,
            # 'env_model_predicted_state': env_model_predicted_state,
            'ctrl_hidden_1':ctrl_hidden_1, 'ctrl_init_1':ctrl_init_1,
            'ctrl_hidden_2': ctrl_hidden_2, 'ctrl_init_2': ctrl_init_2,
            'ctrl_hidden_3': ctrl_hidden_3, 'ctrl_init_3': ctrl_init_3,
            # 'x_encoded': x_encoded
            # 'x_decoded': x_decoded
            }