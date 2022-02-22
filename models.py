#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def get_initializer_stddev(input_shape):
  scale = 1 / np.prod(input_shape)

def TemplatePairStack(c_t, key_dim = 64, num_head = 4, value_dim = 64, num_block = 2, rate = 0.25):
  pair_act = tf.keras.Input((None, c_t)); # pair_act.shape = (N_res, N_res, c_t)
  pair_mask = tf.keras.Input((None, )); # pair_mask.shape = (N_res, N_res)
  pair_act_results = pair_act;
  pair_mask_results = pair_mask;
  for i in range(num_block):
    # triangle_attention_starting_node
    skip = pair_act_results;
    residual = TriangleAttention(c_t, key_dim = key_dim, num_head = num_head, value_dim = value_dim, orientation = 'per_row')([pair_act_results, pair_mask_results]); # pair_act_results.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x: tf.nn.dropout(x, rate = rate, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])))(residual); # pair_act_results.shape = (N_res, N_res, c_t)
    pair_act_results = tf.keras.layers.Add()([skip, residual]);
    # triangle_attention_ending_node
    skip = pair_act_results;
    residual = TriangleAttention(c_t, key_dim = key_dim, num_head = num_head, value_dim = value_dim, orientation = 'per_column')([pair_act_results, pair_mask_results]); # pair_act_results.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x: tf.nn.dropout(x, rate = rate, noise_shape = (tf.shape(x)[0], 1, tf.shape(x)[2])))(residual); # pair_act_results.shape = (N_res, N_res, c_t)
    pair_act_results = tf.keras.layers.Add()([skip, residual]);
    # triangle_multiplication_outgoing
    # TODO:

def Attention(output_dim, key_dim = 64, num_head = 4, value_dim = 64, use_nonbatched_bias = False):
  # NOTE: multi head attention: q_data is query, m_data is key, m_data is value
  # NOTE: differences:
  # 1) qk + bias + tf.expand_dims(nonbatched_bias, axis = 0), ordinary attention only calculate qk.
  # 2) output gets through an output gate controlled by query.
  assert key_dim % num_head == 0;
  assert value_dim % num_head == 0;
  assert key_dim == value_dim;
  q_data = tf.keras.Input((None, key_dim)); # q_data.shape = (batch, N_queries, q_channels)
  m_data = tf.keras.Input((None, value_dim)); # m_data.shape = (batch, N_keys, m_channels)
  bias = tf.keras.Input((None, None, None)); # bias.shape = (batch, num_head or 1, N_queries or 1, N_keys)
  if use_nonbatched_bias:
    nonbatched_bias = tf.keras.Input((None, None), batch_size = num_head); # nonbatched_bias.shape = (num_head, N_queries, N_keys)
  key_dim = key_dim // num_head;
  value_dim = value_dim // num_head;
  q = tf.keras.layers.Dense(num_head * key_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(q_data); # q.shape = (batch, N_queries, num_head * key_dim);
  q = tf.keras.layers.Reshape((-1, num_head, key_dim))(q); # q.shape = (batch, N_queries, num_head, key_dim)
  k = tf.keras.layers.Dense(num_head * key_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(m_data); # k.shape = (batch, N_keys, num_head * key_dim)
  k = tf.keras.layers.Reshape((-1, num_head, key_dim))(k); # k.shape = (batch, N_keys, num_head, key_dim)
  v = tf.keras.layers.Dense(num_head * value_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(m_data); # v.shape = (batch, N_keys, num_head * value_dim)
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
  output = tf.keras.layers.Dense(output_dim, kernel_initializer = tf.keras.initializers.Constant(0.), bias_initializer = tf.keras.initializers.Constant(0.))(weighted_avg); # output.shape = (batch, N_queries, output_dim)
  return tf.keras.Model(inputs = (q_data, m_data, bias, nonbatched_bias) if use_nonbatched_bias else (q_data, m_data, bias), outputs = output);  

def GlobalAttention(output_dim, key_dim = 64, num_head = 4, value_dim = 64):
  # NOTE: multi head attention sharing a same set of value vectors among different heads: query is q_data, key is m_data, value is m_data
  # NOTE: differences:
  # 1) use an extra mask to get weighted average of query along N_queries dimension, whose shape is reduce to batch x q_channels.
  # 2) query's shape is batch x head_num x key_dim (query_len = 1), key's shape is batch x head_num x key_len x key_dim, therefore, qk's shape is batch x head_num x key_len (query_len = 1).
  # 3) value's shape is batch x key_len x value_dim (head_num = 1), multiple head share a same set of value vectors
  # 4) qk + bias, ordinary attention only calculate qk. bias is controlled by query mask.
  # 5) output gets through an output gate controlled by query.
  assert key_dim == value_dim;
  assert key_dim % num_head == 0;
  assert value_dim % num_head == 0;
  q_data = tf.keras.Input((None, key_dim)); # q_data.shape = (batch, N_queries, q_channels)
  m_data = tf.keras.Input((None, value_dim)); # m_data.shape = (batch, N_keys, m_channels)
  q_mask = tf.keras.Input((None, key_dim)); # q_mask.shape = (batch, N_queries, q_channels or 1)
  key_dim = key_dim // num_head;
  value_dim = value_dim // num_head;
  v = tf.keras.layers.Dense(value_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(m_data); # v.shape = (batch, N_keys, value_dim)
  q_mask_broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[0], tf.shape(x[1]) // tf.shape(x[0])))([q_mask, q_data]); # q_mask_broadcast.shape = (batch, N_queries, q_channels)
  q_avg = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = 1) / (tf.math.reduce_sum(x[1], axis = 1) + 1e-10))([q_data, q_mask_broadcast]); # q_avg.shape = (batch, q_channels)
  q = tf.keras.layers.Dense(num_head * key_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(q_avg); # q.shape = (batch, num_head * key_dim)
  q = tf.keras.layers.Reshape((num_head, key_dim))(q); # q.shape = (batch, num_head, key_dim)
  k = tf.keras.layers.Dense(key_dim, use_bias = False, kernel_initializer = tf.keras.initializers.GlorotUniform())(m_data); # k.shape = (batch, N_keys, key_dim)
  bias = tf.keras.layers.Lambda(lambda x: tf.expand_dims(1e9 * (x - 1.), axis = 1)[:,:,:,0])(q_mask); # bias.shape = (batch, 1, N_queries)
  logits = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0] / tf.math.sqrt(tf.cast(tf.shape(x[0])[-1], dtype = tf.float32)), tf.transpose(x[1], (0, 2, 1))) + x[2])([q,k,bias]); # logits.shape = (batch, num_head, N_queries)
  weights = tf.keras.layers.Softmax()(logits); # weights.shape = (batch, num_head, N_keys)
  weighted_avg = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([weights, v]); # weighted_avg.shape = (batch, num_head, value_dim)
  gate_values = tf.keras.layers.Dense(num_head * value_dim, kernel_initializer = tf.keras.initializers.Constant(0.), bias_initializer = tf.keras.initializers.Constant(1.), activation = tf.keras.activations.sigmoid)(q_data); # gate_values.shape = (batch, N_queries, num_head * value_dim)
  gate_values = tf.keras.layers.Reshape((-1, num_head, value_dim))(gate_values); # gate_values.shape = (batch, N_queries, num_head, value_dim)
  weighted_avg = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0], axis = 1) * x[1])([weighted_avg, gate_values]); # weighted_avg.shape = (batch, N_queries, num_head, value_dim)
  weighted_avg = tf.keras.layers.Reshape((-1, num_head * value_dim))(weighted_avg); # weighted_avg.shape = (batch, N_queries, num_head * value_dim)
  output = tf.keras.layers.Dense(output_dim, kernel_initializer = tf.keras.initializers.Constant(0.), bias_initializer = tf.keras.initializers.Constant(0.))(weighted_avg); # output.shape = (batch, N_queries, output_dim)
  return tf.keras.Model(inputs = (q_data, m_data, q_mask), outputs = output);

def MSARowAttentionWithPairBias(c_m, c_z, key_dim = 64, num_head = 4, value_dim = 64):
  # NOTE: multi head self attention: query is msa_act, key is msa_act, value is msa_act.
  # NOTE: differences
  # 1) use msa_mask to control bias, bias's shape is N_seq(batch) x num_head(1) x N_queries(1) x N_res.
  # 2) use pair_act to control nonbatched_bias, nonbatch_bias's shape is num_head x N_res x N_res
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  pair_act = tf.keras.Input((None, c_z)); # pair_act.shape = (N_res, N_res, c_z)
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask); # bias.shape = (N_seq, num_head = 1, N_queries = 1, N_res)
  msa_act_results = tf.keras.layers.LayerNormalization()(msa_act);
  pair_act_results = tf.keras.layers.LayerNormalization()(pair_act); # pair_act_results.shape = (N_res, N_res, c_z)
  nonbatched_bias = tf.keras.layers.Dense(num_head, use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1./np.sqrt(c_z)))(pair_act_results); # nonbatched_bias.shape = (N_res. N_res, num_head)
  nonbatched_bias = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (2, 0, 1)))(nonbatched_bias); # nonbatched_bias.shape = (num_head, N_res, N_res)
  msa_act_results = Attention(c_m, key_dim = key_dim, num_head = num_head, value_dim = value_dim, use_nonbatched_bias = True)([msa_act_results, msa_act_results, bias, nonbatched_bias]); # msa_act_results.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = (msa_act, msa_mask, pair_act), outputs = msa_act_results);

def MSAColumnAttention(c_m, key_dim = 64, num_head = 4, value_dim = 64):
  # NOTE: multi head self attention: query is msa_act.T, key is msa_act.T, value is msa_act.T.
  # NOTE: differences
  # 1) use msa_mask to control bias, bias's shape is N_res(batch) x num_head(1) x N_queries(1) x N_seq.
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  msa_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_mask_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(msa_mask); # msa_mask_results.shape = (N_res, N_seq)
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask_results); # bias.shape = (N_res, 1, 1, N_seq)
  msa_act_results = tf.keras.layers.LayerNormalization()(msa_act_results); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_act_results = Attention(c_m, key_dim = key_dim, num_head = num_head, value_dim = value_dim, use_nonbatched_bias = False)([msa_act_results, msa_act_results, bias]); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act_results); # msa_act_results.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = (msa_act, msa_mask), outputs = msa_act_results);

def MSAColumnGlobalAttention(c_m, key_dim = 64, num_head = 4, value_dim = 64):
  # NOTE: multi head self attention: query is msa_act.T, key is msa_act.T, value is msa_act.T.
  # NOTE: differences
  # 1) use msa_mask to control q_mask which controls bias in global attention.
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  msa_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_mask_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(msa_mask); # msa_mask_results.shape = (N_res, N_seq)
  #bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask_results); # bias.shape = (N_res, 1, 1, N_seq)
  msa_act_results = tf.keras.layers.LayerNormalization()(msa_act_results); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_mask_results = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(msa_mask_results); # msa_mask_results.shape = (N_res, N_seq, 1)
  msa_act_results = GlobalAttention(c_m, key_dim = key_dim, num_head = num_head, value_dim = value_dim)([msa_act_results, msa_act_results, msa_mask_results]); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act_results); # msa_act_results.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = (msa_act, msa_mask), outputs = msa_act_results);

def TriangleAttention(c_z, key_dim = 64, num_head = 4, value_dim = 64, orientation = 'per_column'):
  # NOTE: multi head self attention: query is pair_act, key is pair_act, value is pair_act.
  # NOTE: difference:
  # 1) use pair_mask to control bias.
  # 2) use pair_act to control nonbatched_bias.
  assert orientation in ['per_column', 'per_row'];
  pair_act = tf.keras.Input((None, c_z)); # pair_act.shape = (N_res, N_res, c_z)
  pair_mask = tf.keras.Input((None,)); # pair_mask.shape = (N_res, N_res)
  if orientation == 'per_column':
    pair_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(pair_act); # pair_act_results.shape = (N_res, N_res, c_z)
    pair_mask_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(pair_mask); # pair_mask_results.shape = (N_res, N_res)
  else:
    pair_act_results = pair_act;
    pair_mask_results = pair_mask;
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(pair_mask); # bias.shape = (N_seq, 1, 1, N_res) if per_row else (N_res, 1, 1, N_seq)
  pair_act_results = tf.keras.layers.LayerNormalization()(pair_act_results); # pair_act_results.shape = (N_res, N_res, c_z)
  nonbatched_bias = tf.keras.layers.Dense(num_head, use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1./np.sqrt(c_z)))(pair_act_results); # nonbatched_bias.shape = (N_res, N_res, num_head)
  nonbatched_bias = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (2, 0, 1)))(nonbatched_bias); # nonbatched_bias.shape = (num_head, N_res, N_res)
  pair_act_results = Attention(c_z, key_dim = key_dim, num_head = num_head, value_dim = value_dim, use_nonbatched_bias = True)([pair_act_results, pair_act_results, bias, nonbatched_bias]); # pair_act_results.shape = (N_res, N_res, c_z)
  if orientation == 'per_column':
    pair_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(pair_act_results); # pair_act_results.shape = (N_res, N_res, c_z)
  return tf.keras.Model(inputs = (pair_act, pair_mask), outputs = pair_act_results);

def TriangleMultiplication(c_z, intermediate_channel = 64, mode = 'outgoing'):
  assert mode in ['outgoing', 'incoming'];
  act = tf.keras.Input((None, c_z)); # act.shape = (N_res, N_res, c_z)
  mask = tf.keras.Input((None,)); # mask.shape = (N_res, N_res)
  mask_results = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(mask); # mask_results.shape = (N_res, N_res, 1)
  act_results = tf.keras.layers.LayerNormalization()(act); # act_results.shape = (N_res, N_res, c_z)
  input_act = act_results;
  # left projection
  left_projection = tf.keras.layers.Dense(intermediate_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act_results); # left_projection.shape = (N_res, N_res, intermediate_channel)
  left_proj_act = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask_results, left_projection]); # left_proj_act.shape = (N_res, N_res, intermediate_channel)
  # right projection
  right_projection = tf.keras.layers.Dense(intermediate_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act_results); # right_projection.shape = (N_res, N_res, intermediate_channel)
  right_proj_act = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask_results, right_projection]); # right_proj_act.shape = (N_res, N_res, intermediate_channel)
  # left gate
  left_gate_values = tf.keras.layers.Dense(intermediate_channel, activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Constant(1.))(act_results); # left_gate_values.shape = (N_res, N_res, intermediate_channel)
  right_gate_values = tf.keras.layers.Dense(intermediate_channel, activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Constant(1.))(act_results); # right_gate_values.shape = (N_res, N_res, intermediate_channel)
  # gate projection
  left_proj_act = tf.keras.layers.Multiply()([left_proj_act, left_gate_values]); # left_proj_act.shape = (N_res, N_res, intermediate_channel)
  right_proj_act = tf.keras.layers.Multiply()([right_proj_act, right_gate_values]); # right_proj_act.shape = (N_res, N_res, intermediate_channel)
  # apply equation
  if mode == 'outgoing':
    act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(tf.transpose(x[0], (2,0,1)), tf.transpose(x[1], (2,0,1)), transpose_b = True), (1,2,0)))([left_proj_act, right_proj_act]); # act_results.shape = (N_res, N_res, intermediate_channel)
  else:
    act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(tf.transpose(x[0], (2,1,0)), tf.transpose(x[1], (2,1,0)), transpose_b = True), (2,1,0)))([left_proj_act, right_proj_act]); # act_results.shape = (N_res, N_res, intermediate_channel)
  act_results = tf.keras.layers.LayerNormalization()(act_results); # act_results.shape = (N_res, N_res, intermediate_channel)
  act_results = tf.keras.layers.Dense(c_z, kernel_initializer = tf.keras.initializers.Zeros())(act_results); # act_results.shape = (N_res, N_res, c_z)
  gate_values = tf.keras.layers.Dense(c_z, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Constant(1.))(input_act);
  act_results = tf.keras.layers.Multiply()([act_results, gate_values]); # act_results.shape = (N_res, N_res, c_z)
  return tf.keras.Model(inputs = (act, mask), outputs = act_results);

def MaskedMsaHead(c_m):
  msa = tf.keras.Input((None, c_m)); # msa.shape = (N_seq, N_seq, c_m)
  

if __name__ == "__main__":
  import numpy as np;
  q_data = np.random.normal(size = (4, 20, 64));
  m_data = np.random.normal(size = (4, 10, 64));
  bias = np.random.normal(size = (4, 1, 1, 10));
  results = Attention(100)([q_data, m_data, bias]);
  print(results.shape);
  q_data = np.random.normal(size = (4, 20, 64));
  m_data = np.random.normal(size = (4, 20, 64));
  q_mask = np.random.randint(low = 0, high = 2, size = (4, 20, 64));
  results = GlobalAttention(100)([q_data, m_data, q_mask]);
  print(results.shape);
  msa_act = np.random.normal(size = (4, 20, 64));
  msa_mask = np.random.randint(low = 0, high = 2, size = (4, 20));
  pair_act = np.random.normal(size = (20, 20, 32));
  results = MSARowAttentionWithPairBias(64, 32)([msa_act, msa_mask, pair_act]);
  print(results.shape);
  results = MSAColumnAttention(64)([msa_act, msa_mask]);
  print(results.shape);
  results = MSAColumnGlobalAttention(64)([msa_act, msa_mask]);
  print(results.shape);
  pair_act = np.random.normal(size = (20, 20, 64));
  pair_mask = np.random.randint(low = 0, high = 2, size = (20, 20));
  results = TriangleAttention(64)([pair_act, pair_mask]);
  print(results.shape);
  results = TriangleMultiplication(64, mode = 'outgoing')([pair_act, pair_mask]);
  print(results.shape);
  results = TriangleMultiplication(64, mode = 'incoming')([pair_act, pair_mask]);
  print(results.shape);
