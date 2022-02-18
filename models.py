#!/usr/bin/python3

import tensorflow as tf;

def Attention(key_dim = 64, num_head = 4, value_dim = 64, use_nonbatched_bias = False):
  assert key_dim % num_head == 0;
  assert value_dim % num_head == 0;
  key_dim = key_dim // num_head;
  value_dim = value_dim // num_head;
  q_data = tf.keras.Input((None, None)); # q_data.shape = (batch, N_queries, q_channels)
  m_data = tf.keras.Input((None, None)); # m_data.shape = (batch, N_keys, m_channels)
  bias = tf.keras.Input((None, None)); # bias.shape = (batch, N_queries, N_keys)
  if use_nonbatched_bias:
    nonbatched_bias = tf.keras.Input((None,)); # nonbatched_bias.shape = (N_queries, N_keys)
  q = tf.keras.layers.Dense(num_head * key_dim, use_bias = False)(q_data); # q.shape = (batch, N_queries, num_head * key_dim);
  q = tf.keras.layers.Reshape((-1, num_head, key_dim))(q); # q.shape = (batch, N_queries, num_head, key_dim)
  k = tf.keras.layers.Dense(num_head * key_dim, use_bias = False)(m_data); # k.shape = (batch, N_keys, num_head * key_dim)
  k = tf.keras.layers.Reshape((-1, num_head, key_dim))(k); # k.shape = (batch, N_keys, num_head, key_dim)
  v = tf.keras.layers.Dense(num_head * key_dim, use_bias = False)(m_data); # v.shape = (batch, N_keys, num_head * value_dim)
  v = tf.keras.layers.Reshape((-1, num_head, value_dim))(v); # v.shape = (batch, N_keys, num_head, value_dim)
  logits = tf.keras.layers.Lambda(lambda x, k: tf.linalg.matmul(tf.transpose(x[0], (0, 2, 1, 3)) / tf.math.sqrt(k), tf.transpose(x[1], (0, 2, 1, 3)), transpose_b = True) + tf.expand_dims(x[2], axis = 1), arguments = {'k': key_dim})([q, k, bias]); # logits.shape = (batch, num_head, N_queries, N_keys)
  if use_nonbatched_bias:
    logits = tf.keras.layers.Lambda(lambda x: x[0] + tf.reshape(x[1], (1,1,tf.shape(x[1])[0],tf.shape(x[1])[1])))([logits, nonbatched_bias]); # logits.shape = (batch, num_head, N_queries, N_keys)
  weights = tf.keras.layers.Softmax()(logits); # weights.shape = (batch, num_head, N_queries, N_keys)
  weighted_avg = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(x[0], tf.transpose(x[1], (0, 2, 1, 3))), (0, 2, 1, 3)))([weights, v]); # weighted_avg.shape = (batch, N_queryeis, num_head, value_dim)
  

def MSARowAttentionWithPairBias():
  msa_act = tf.keras.Input((None, None)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  pair_act = tf.keras.Input((None, None)); # pair_act.shape = (N_res, N_res, c_z)
  msa_act_results = tf.keras.layers.LayerNormalization()(msa_act);
  pair_act_results = tf.keras.layers.LayerNormalization()(pair_act);
  
