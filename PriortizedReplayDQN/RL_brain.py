"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):
  """
  This SumTree code is modified version and the original code is from: 
  https://github.com/jaara/AI-blog/blob/master/SumTree.py
  
  Story the data with it priority in tree and data frameworks.
  """
  data_pointer = 0

  def __init__(self, capacity):
    self.capacity = capacity # for all priority values
    self.tree = np.zeros(2*capacity -1)
    #[=================부모 노드들==========][======선험정보들이 기록될 곳 ======]
    self.data = np.zeros(capacity, dtype = object)
    #[=========data frame========]
    #    size : capacity

  def add_new_priority(self, p, data):
    leaf_idx = self.data_pointer + self.capacity -1

    self.data[self.data_pointer] = data #data_fram update
    self.update(leaf_idx, p) # tree_frame update

    self.data_pointer += 1
    if self.data_pointer >= self.capacity: # replace when exceed the capacity
      self.data_pointer=0

  def update(self, tree_idx, p):
    change = p-self.tree[tree_idx]

    self.tree[tree_idx] = p
    self._propagate_change(tree_idx, change)

  def _propagate_change(self, tree_idx, change):
    """ change the sum of priority value in all parent nodes"""
    parent_idx = (tree_idx - 1) // 2
    self.tree[parent_idx] += change
    if parent_idx != 0:
      self._propagate_change(parent_idx, change)

  def get_leaf(self, lower_bound):
    leaf_idx = self._retrieve(lower_bound) # search the max leaf priority base on the lower_bound
    data_idx = leaf_idx - self.capacity + 1
    return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

  def _retrieve(self, lower_bound, parent_idx=0):
    """
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
   
    Array type for storing:
    [0,1,2,3,4,5,6]
    """
    left_child_idx = 2 * parent_idx + 1
    right_child_idx = left_child_idx + 1

    if left_child_idx >= len(self.tree):
      return parent_idx

    if self.tree[left_child_idx] == self.tree[right_child_idx]:
      return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
    if lower_bound <= self.tree[left_child_idx]: # downward search, always search for a higher priority node
      return self._retrieve(lower_bound, left_child_idx)
    else:
      return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
      return self.tree[0] # the root

class Memory(object): # stored as ( s, a, r, s_ ) in SumTree
  """
  This SumTree code is modified version and the original code is from:
  https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
  """
  epsilon = 0.01 # small amount to avoid zero priority
  alpha = 0.6 # [0~1] convert the importance of TD error to priority
  beta = 0.4  # importance-sampling, from initial value increasing to 1
  beta_increment_per_sampling = 0.001
  abs_err_upper = 1. # clipped abs error

  def __init__(self, capacity):
    self.tree = SumTree(capacity)

  def store(self, transition):
    max_p = np.max(self.tree.tree[-self.tree.capacity:])
    if max_p = 0:
      max_p = self.abs_err_upper
    self.tree.add_new_priority(max_p, transition) # set the max p for new p

  def sample(self, n):
    batch_idx, batch_memory, ISWeight = [],[],[]
    segment = self.tree.root_priority / n
    self.beta = np.min([1, self.beta + self.bet_icrement_per_sapling]) # max 1

    min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
    maxiwi = np.power(self.tree.capacity * min_prob, -self.beta) # for later normalizing ISWeights

    for i in range(n):
      a = segment * i
      b = segment * (i + 1)
      lower_bound = np.random.uniform(a, b)
      idx, p, data = self.tree.get_leaf(lower_bound)
      prob = p/self.tree.root_priority
      ISWeights.append(self.tree.capacity * prob)
      batch_idx.append(idx)
      batch_memory.append(data)

    ISWeights = np.vstack(ISWeights)
    ISWeights = np.power(ISWeights, -self.beta) /maxiwi # normalize
    return batch_idx, np.vstack(batch_memory), ISWeights

  def update(self, idx, error):
    p = self._get_priority(error)
    self.tree.update(idx, p)

  def _get_priority(self, error):
    error += self.epsilon # avoid 0
    clipped_error = np.clip(error, 0, self.abs_err_upper)
    return np.power(clipped_error, self.alpha)

class DQNPrioritizedReplay:
  def __init__(
    self,
    n_actions,
    n_features,
    learning_rate=0.005,
    reward_decay=0.9,
    e_greedy=0.9
    replace_targetr_iter=500,
    memory_size=10000,
    batch_size=32,
    e_greedy_increment=None,
    output_graph=False,
    prioritized=True,
    sess=None,
    ):
    self.n_actions = n_actions
    self.n_features = n_features
    self.lr = learning_rate
    slef.gamma = reward_decay
    self.epsilon_max = e_greedy
    self.replace_target_iter = replace_target_iter
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.epsilon_increment = e_greedy_increment
    self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

    self.prioritized = prioritized #decide to use double q or not

    self.learn_step_counter = 0

    self._build_net()

    if self.prioritized:
      self.memory = Memory(capacity=memory_size)
    else:
      self.memory = np.zeros((self.memory_size, n_features*2+2))

    if sess is None:
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
    else:
      self.sess = sess

    if output_graph:
      tf.summary.FileWriter("logs/", self.sess.graph)

    self.cost_his = []

  def _build_net(self):
    def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
      with tf.variable_scope('l1'):
        w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=cnames)
        b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, colloections=c_names)
        l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

      with tf.variable_scope('l2'):
