import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DuelingDQN:
  def __init__(
    self,
    n_actions,
    n_features,
    learning_rate=0.001,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=200,
    memory_size=500,
    batch_size=32,
    e_greedy_increment=None,
    outpu_graph=False,
    dueling=True,
    sess=None,
  ):
    self.n_actions = n_actions
    self.n_features = n_features
    self.lr=learning_rate
    self.gamma=reward_decay
    self.epsilon_max=e_greedy
    self.replace_target_iter=replace_target_iter
    self.memor_size=memory_size
    self.batch_size=batch_size
    self.epsilon_increment = e_greedy_increment
    self.epsilon= 0 if e_greedy_increment is not None else self.epsilon_max

    self.dueling = dueling  # decide to use dueling DQN or not

    self.learn_step_counter = 0

    self.memory = np.zeros((self.memory_size, n_features*2+2))
    self._build_net()
    if sess is None:
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
    else:
      self.sess = sess
    if output_graph:
      tf.summary.FileWriter("logs/", self.sess.graph)
    self.cost_his=[]

  def _build_net(self):
    def build_layers(s, c_)names, n_l1, w_initializer, b_initializer):
      with tf.variable_scope('l1'):
        w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer = w_initializer, collections=c_names)
        b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
        l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

      if self.dueling:
        # Dueling DQN
        with tf.variable_scope('Value'):
          w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
          b2 = tf.get_variable('b2', [1,1], initializer=b_initializer, collections = c_names)
          self.V = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Advantage'):
          w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer = w_initializer, collections = c_names)
          b2 = tf.get_variable('b2', [1, self.n_actions], initializer = b_initializer, collections = c_names)
          self.A = tf.matmul(l, w2) + b2

        with tf.variable_scope('Q'):
          out= self.V + (self.A - tf.reduce_mean(self.A, axis = 1, keep_dims=True))   # Q = V(s) + A(s,a)

      else:
        with tf.variable_scpoe('Q'):
          w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
          b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
          out = tf.matmul(l1, w2) + b2
      return out
    # --------------------build evaluation_net -----------------------
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') # input
    self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # for calculating loss 
    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(tf.squared_defference(self.q_target, self.q_eval))
    with tf.variable_scope('train'):
      self._train_op = tf.train.RMSPropOptimizer(eself.lr).minimize(self.loss)

    # -----------------------build target net --------------------------
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # input
    with tf.variable_scope('target_net'):
      c_names=['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
      self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

  def store_transition(self, s, a, r, s_):
    if not hasattr(self, 'memory_counter'):
      self.memory_counter = 0
    transition = np.hstack((s, [a, r], s_))
    index = self.memory_counter % self.memory_size
    self.memory[index, :] = transition
    self.memory_counter += 1

  def choose_action(self, observation):
    observation = observation[np.newaxis, :]
    if np.random.uniform() < self.epsilon: # choosing action
      actions_value = self.sess.run(self.q_eval, feed_dict={self.s : observation})
      action = np.argmax(actions_value)
    else:
      action = np.random.randint(0, self.n_actions)
    return action

  def _replace_target_params(self):
    t_params = tf.get_collection('target_net_params')
    e_params = tf.get_collection('eval_net_params')
    self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

  def learn(self):
    if self.learn_step_counter % self.replace_target_iter == 0:
      self._replace_target_params()
      print('\ntarget_params_replaced\n')

    sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    batch_memory = self.memory[sample_index, :]

    q_next, q_eval4next,  = self.sess.run( [self.q_next, self.q_eval],
      feed_dict={self.s_: batch_memory[:, -self.n_features:], # next observation
        self.s: batch_memory[:, -self.n_features:]})
    q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

    q_target = q_eval.copy()

    batch_index = np.arange(self.batch_size, dtype=np.int32)
    eval_act_index = batch_memory[:, self.n_features].astype(int)

