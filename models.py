#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def Attention(output_dim, key_dim = 64, num_head = 4, value_dim = 64, use_nonbatched_bias = False):
  assert key_dim % num_head == 0;
  assert value_dim % num_head == 0;
  q_data = tf.keras.Input((None, key_dim)); # q_data.shape = (batch, N_queries, q_channels)
  m_data = tf.keras.Input((None, key_dim)); # m_data.shape = (batch, N_keys, m_channels)
  bias = tf.keras.Input((num_head, None, None)); # bias.shape = (batch, num_head, N_queries, N_keys)
  if use_nonbatched_bias:
    nonbatched_bias = tf.keras.Input((None, None), batch = num_head); # nonbatched_bias.shape = (num_head, N_queries, N_keys)
  key_dim = key_dim // num_head;
  value_dim = value_dim // num_head;
  q = tf.keras.layers.Dense(num_head * key_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(q_data); # q.shape = (batch, N_queries, num_head * key_dim);
  q = tf.keras.layers.Reshape((-1, num_head, key_dim))(q); # q.shape = (batch, N_queries, num_head, key_dim)
  k = tf.keras.layers.Dense(num_head * key_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(m_data); # k.shape = (batch, N_keys, num_head * key_dim)
  k = tf.keras.layers.Reshape((-1, num_head, key_dim))(k); # k.shape = (batch, N_keys, num_head, key_dim)
  v = tf.keras.layers.Dense(num_head * key_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(m_data); # v.shape = (batch, N_keys, num_head * value_dim)
  v = tf.keras.layers.Reshape((-1, num_head, value_dim))(v); # v.shape = (batch, N_keys, num_head, value_dim)
  logits = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(tf.transpose(x[0], (0, 2, 1, 3)) / tf.math.sqrt(tf.cast(tf.shape(x[0])[-1], dtype = tf.float32)), tf.transpose(x[1], (0, 2, 1, 3)), transpose_b = True) + x[2])([q, k, bias]); # logits.shape = (batch, num_head, N_queries, N_keys)
  if use_nonbatched_bias:
    logits = tf.keras.layers.Lambda(lambda x: x[0] + tf.expand_dims(x[1], axis = 0))([logits, nonbatched_bias]); # logits.shape = (batch, num_head, N_queries, N_keys)
  weights = tf.keras.layers.Softmax()(logits); # weights.shape = (batch, num_head, N_queries, N_keys)
  weighted_avg = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(x[0], tf.transpose(x[1], (0, 2, 1, 3))), (0, 2, 1, 3)))([weights, v]); # weighted_avg.shape = (batch, N_queryeis, num_head, value_dim)
  gate_values = tf.keras.layers.Dense(num_head * value_dim, kernel_initializer = tf.keras.initializers.Constant(0.), bias_initializer = tf.keras.initializers.Constant(1.), activation = tf.keras.activations.sigmoid)(q_data); # gate_values.shape = (batch, N_queries, num_head * value_dim)
  gate_values = tf.keras.layers.Reshape((-1, num_head, value_dim))(gate_values); # gate_values.shape = (batch, N_queries, num_head, value_dim)
  weighted_avg = tf.keras.layers.Multiply()([weighted_avg, gate_values]); # weighted_avg.shape = (batch, N_queries, num_head, value_dim)
  weighted_avg = tf.keras.layers.Reshape((-1, num_head * value_dim))(weighted_avg); # weighted_avg.shape = (batch, N_queries, num_head * value_dim)
  output = tf.keras.layers.Dense(output_dim, kernel_initializer = tf.keras.initializers.GlorotUniform())(weighted_avg); # output.shape = (batch, N_queries, output_dim)
  return tf.keras.Model(inputs = (q_data, m_data, bias, nonbatched_bias) if use_nonbatched_bias else (q_data, m_data, bias), outputs = output);

def MSARowAttentionWithPairBias(c_m, c_z, key_dim = 64, num_head = 4, value_dim = 64):
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  pair_act = tf.keras.Input((None, c_z)); # pair_act.shape = (N_res, N_res, c_z)
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask); # bias.shape = (N_seq, 1, 1, N_res)
  msa_act_results = tf.keras.layers.LayerNormalization()(msa_act);
  pair_act_results = tf.keras.layers.LayerNormalization()(pair_act); # pair_act_results.shape = (N_res, N_res, c_z)
  nonbatched_bias = tf.keras.layers.Dense(num_head, use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1./np.sqrt(c_z)))(pair_act_results); # nonbatched_bias.shape = (N_res. N_res, num_head)
  nonbatched_bias = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (2, 0, 1)))(nonbatched_bias); # nonbatched_bias.shape = (num_head, N_res, N_res)
  msa_act_results = Attention(c_m, key_dim = key_dim, num_head = num_head, value_dim = value_dim, use_nonbatched_bias = True)([msa_act_results, msa_act_results, bias, nonbatched_bias]); # msa_act_results.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = (msa_act, msa_mask, pair_act), outputs = msa_act_results);

if __name__ == "__main__":
  import numpy as np;
  q_data = np.random.normal(size = (4, 20, 64));
  m_data = np.random.normal(size = (4, 10, 64));
  bias = np.random.normal(size = (4, 20, 10));
  results = Attention(100)([q_data, m_data, bias]);
  print(results.shape);
  msa_act = np.random.normal(size = (4, 20, 64));
  msa_mask = np.random.randint(low = 0, high = 1, size = (4, 20));
  pair_act = np.random.normal(size = (4, 20, 32));
  results = MSARowAttentionWithPairBias(64, 32)([msa_act, msa_mask, pair_act]);
  print(results.shape);
