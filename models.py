#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from residue_constants import restype_order, atom_order, atom_type_num;

def TemplatePairStack(c_t, num_head = 4, num_intermediate_channel = 64, num_block = 2, rate = 0.25, **kwargs):
  pair_act = tf.keras.Input((None, c_t)); # pair_act.shape = (N_res, N_res, c_t)
  pair_mask = tf.keras.Input((None, )); # pair_mask.shape = (N_res, N_res)
  pair_act_results = pair_act;
  pair_mask_results = pair_mask;
  for i in range(num_block):
    # triangle_attention_starting_node
    residual = TriangleAttention(c_t, num_head = num_head, orientation = 'per_row', name = 'block%d/triangle_attention_starting_node' % i)([pair_act_results, pair_mask_results]); # pair_act_results.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': rate})(residual); # pair_act_results.shape = (N_res, N_res, c_t)
    pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]);
    # triangle_attention_ending_node
    residual = TriangleAttention(c_t, num_head = num_head, orientation = 'per_column', name = 'block%d/triangle_attention_ending_node' % i)([pair_act_results, pair_mask_results]); # pair_act_results.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (tf.shape(x)[0], 1, tf.shape(x)[2])), arguments = {'r': rate})(residual); # pair_act_results.shape = (N_res, N_res, c_t)
    pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]);
    # triangle_multiplication_outgoing
    residual = TriangleMultiplication(c_t, intermediate_channel = num_intermediate_channel, mode = 'outgoing', name = 'block%d/triangle_multiplication_outgoing' % i)([pair_act_results, pair_mask_results]); # residual.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': rate})(residual); # residual.shape = (N_res, N_res, c_t)
    pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]);
    # triangle_multiplication_incoming
    residual = TriangleMultiplication(c_t, intermediate_channel = num_intermediate_channel, mode = 'incoming', name = 'block%d/triangle_multiplication_incoming' % i)([pair_act_results, pair_mask_results]); # residual.shape = (N_res, N_res, act)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': rate})(residual); # residual.shape = (N_res, N_res, c_t)
    pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]);
  return tf.keras.Model(inputs = (pair_act, pair_mask), outputs = pair_act_results, **kwargs);

def Transition(c_t, num_intermediate_factor = 4):
  act = tf.keras.Input((None, c_t)); # act.shape = (batch, N_res, c_t)
  mask = tf.keras.Input((None,)); # mask.shape = (batch, N_res)
  mask_results = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(mask); # mask_results.shape = (batch, N_res, 1)
  act_results = tf.keras.layers.LayerNormalization()(act); # act_results.shape = (batch, N_res, c_t)
  act_results = tf.keras.layers.Dense(c_t * num_intermediate_factor, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = np.sqrt(2)))(act_results); # act_results.shape = (batch, N_res, 4*c_t)
  act_results = tf.keras.layers.Dense(c_t, kernel_initializer = tf.keras.initializers.Zeros())(act_results); # act_results.shape = (batch, N_res, c_t)
  return tf.keras.Model(inputs = (act, mask), outputs = act_results);

def Attention(output_dim, key_dim = 64, num_head = 4, value_dim = 64, use_nonbatched_bias = False, **kwargs):
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
  return tf.keras.Model(inputs = (q_data, m_data, bias, nonbatched_bias) if use_nonbatched_bias else (q_data, m_data, bias), outputs = output, **kwargs);  

def GlobalAttention(output_dim, key_dim = 64, num_head = 4, value_dim = 64, **kwargs):
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
  return tf.keras.Model(inputs = (q_data, m_data, q_mask), outputs = output, **kwargs);

def MSARowAttentionWithPairBias(c_m, c_z, num_head = 4, **kwargs):
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
  msa_act_results = Attention(c_m, key_dim = c_m, num_head = num_head, value_dim = c_m, use_nonbatched_bias = True)([msa_act_results, msa_act_results, bias, nonbatched_bias]); # msa_act_results.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = (msa_act, msa_mask, pair_act), outputs = msa_act_results, **kwargs);

def MSAColumnAttention(c_m, num_head = 4, **kwargs):
  # NOTE: multi head self attention: query is msa_act.T, key is msa_act.T, value is msa_act.T.
  # NOTE: differences
  # 1) use msa_mask to control bias, bias's shape is N_res(batch) x num_head(1) x N_queries(1) x N_seq.
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  msa_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_mask_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(msa_mask); # msa_mask_results.shape = (N_res, N_seq)
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask_results); # bias.shape = (N_res, 1, 1, N_seq)
  msa_act_results = tf.keras.layers.LayerNormalization()(msa_act_results); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_act_results = Attention(c_m, key_dim = c_m, num_head = num_head, value_dim = c_m, use_nonbatched_bias = False)([msa_act_results, msa_act_results, bias]); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act_results); # msa_act_results.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = (msa_act, msa_mask), outputs = msa_act_results, **kwargs);

def MSAColumnGlobalAttention(c_m, num_head = 4, **kwargs):
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
  msa_act_results = GlobalAttention(c_m, key_dim = c_m, num_head = num_head, value_dim = c_m)([msa_act_results, msa_act_results, msa_mask_results]); # msa_act_results.shape = (N_res, N_seq, c_m)
  msa_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act_results); # msa_act_results.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = (msa_act, msa_mask), outputs = msa_act_results, **kwargs);

def TriangleAttention(c_z, num_head = 4, orientation = 'per_column', **kwargs):
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
  pair_act_results = Attention(c_z, key_dim = c_z, num_head = num_head, value_dim = c_z, use_nonbatched_bias = True)([pair_act_results, pair_act_results, bias, nonbatched_bias]); # pair_act_results.shape = (N_res, N_res, c_z)
  if orientation == 'per_column':
    pair_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(pair_act_results); # pair_act_results.shape = (N_res, N_res, c_z)
  return tf.keras.Model(inputs = (pair_act, pair_mask), outputs = pair_act_results, **kwargs);

def TriangleMultiplication(c_z, intermediate_channel = 64, mode = 'outgoing', **kwargs):
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
  return tf.keras.Model(inputs = (act, mask), outputs = act_results, **kwargs);

def MaskedMsaHead(c_m, num_output = 23, **kwargs):
  msa = tf.keras.Input((None, c_m)); # msa.shape = (N_seq, N_seq, c_m)
  logits = tf.keras.layers.Dense(num_output, kernel_initializer = tf.keras.initializers.Zeros())(msa);
  return tf.keras.Model(inputs = msa, outputs = logits, **kwargs);

def PredictedLDDTHead(c_s, num_channels = 128, num_bins = 50, **kwargs):
  act = tf.keras.Input((c_s,)); # act.shape = (N_res, c_s)
  act_results = tf.keras.layers.LayerNormalization()(act); # act_results.shape = (N_res, c_s)
  act_results = tf.keras.layers.Dense(num_channels, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = np.sqrt(2)))(act_results);
  act_results = tf.keras.layers.Dense(num_channels, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = np.sqrt(2)))(act_results);
  logits = tf.keras.layers.Dense(num_bins, kernel_initializer = tf.keras.initializers.Zeros())(act_results);
  return tf.keras.Model(inputs = act, outputs = logits, **kwargs);

def PredictedAlignedErrorHead(c_z, num_bins = 64, max_error_bin = 31):
  act = tf.keras.Input((None, c_z)); # act.shape = (N_res, N_res, c_z)
  logits = tf.keras.layers.Dense(num_bins, kernel_initializer = tf.keras.initializers.Zeros())(act);
  breaks = tf.keras.layers.Lambda(lambda x, m, n: tf.linspace(0., m, n - 1), arguments = {'m': max_error_bin, 'n': num_bins})(act);
  return tf.keras.Model(inputs = act, outputs = (logits, breaks));

def ExperimentallyResolvedHead(c_s):
  single = tf.keras.Input((c_s,)); # single.shape = (N_res, c_s)
  logits = tf.keras.layers.Dense(atom_type_num, kernel_initializer = tf.keras.initializers.Zeros())(single);
  return tf.keras.Model(inputs = single, outputs = logits);

def DistogramHead(c_z, num_bins = 64, first_break = 2.3125, last_break = 21.6875):
  pair = tf.keras.Input((None, c_z)); # pair.shape = (N_res, N_res, c_z)
  half_logits = tf.keras.layers.Dense(num_bins, kernel_initializer = tf.keras.initializers.Zeros())(pair);
  logits = tf.keras.layers.Lambda(lambda x: x + tf.transpose(x, (1,0,2)))(half_logits);
  breaks = tf.keras.layers.Lambda(lambda x, f, l, n: tf.linspace(f, l, n - 1), arguments = {'f': first_break, 'l': last_break, 'n': num_bins})(pair);
  return tf.keras.Model(inputs = pair, outputs = (logits, breaks));

def OuterProductMean(num_output_channel, c_m, num_outer_channel = 32):
  act = tf.keras.Input((None, c_m)); # act.shape = (N_seq, N_res, c_m)
  mask = tf.keras.Input((None,)); # mask.shape = (N_seq, N_res)
  mask_results = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(mask); # mask_results.shape = (N_seq, N_res, 1)
  act_results = tf.keras.layers.LayerNormalization()(act); # act_results.shape = (N_seq, N_res, c_m)
  left_act_results = tf.keras.layers.Dense(num_outer_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act_results); # left_act_results.shape = (N_seq, N_res, num_outer_channel)
  left_act_results = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask_results, left_act_results]); # left_act_results.shape = (N_seq, N_res, num_outer_channel)
  right_act_results = tf.keras.layers.Dense(num_outer_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act_results); # right_act_results.shape = (N_seq, N_res, num_outer_channel)
  right_act_results = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask_results, right_act_results]); # right_act_results.shape = (N_seq, N_res, num_outer_channel)
  left_act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(left_act_results); # left_act_results.shape = (N_seq, num_outer_channel, N_res)
  act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.reshape(tf.linalg.matmul(tf.reshape(x[0], (tf.shape(x[0])[0], -1)), tf.reshape(x[1], (tf.shape(x[1])[0], -1)), transpose_a = True), (tf.shape(x[0])[1], tf.shape(x[0])[2], tf.shape(x[1])[1], tf.shape(x[1])[2])), (2,1,0,3)))([left_act_results, right_act_results]); # act_results.shape = (N_res, N_res, num_outer_channel, num_outer_channel)
  act_results_reshape = tf.keras.layers.Lambda(lambda x, n: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], n**2)), arguments = {'n': num_outer_channel})(act_results); # act_results.shape = (N_res, N_res, num_outer_channel**2)
  act_results = tf.keras.layers.Dense(num_output_channel, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(act_results_reshape); # act_results.shape = (N_res, N_res, num_output_channel)
  act_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(act_results); # act_results.shape = (N_res, N_res, num_output_channel)
  norm = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(tf.transpose(x, (2,1,0)), tf.transpose(x,(2,1,0)), transpose_b = True), (1,2,0)))(mask_results); # norm.shape = (N_res, N_res, 1)
  act_results = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + 1e-3))([act_results, norm]); # act_results.shape = (N_res, N_res, num_output_channel)
  return tf.keras.Model(inputs = (act, mask), outputs = act_results);

def dgram_from_positions(min_bin, max_bin, num_bins = 39):
  positions = tf.keras.Input((3,)); # positions.shape = (N_res, 3)
  lower_breaks = tf.keras.layers.Lambda(lambda x,l,u,n: tf.linspace(l,u,n), arguments = {'l': min_bin, 'u': max_bin, 'n': num_bins})(positions); # lower_breaks.shape = (num_bins)
  lower_breaks = tf.keras.layers.Lambda(lambda x: tf.math.square(x))(lower_breaks); # lower_breaks.shape = (num_bins,)
  upper_breaks = tf.keras.layers.Lambda(lambda x: tf.concat([x[1:], [1e8]], axis = -1))(lower_breaks); # upper_breaks.shape = (num_bins,)
  dist2 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.square(tf.expand_dims(x, axis = -2) - tf.expand_dims(x, axis = -3)), axis = -1, keepdims = True))(positions); # dist2.shape = (N_res, N_res, 1)
  dgram = tf.keras.layers.Lambda(lambda x: tf.cast(x[0] > x[1], dtype = tf.float32) * tf.cast(x[0] < x[2], dtype = tf.float32))([dist2, lower_breaks, upper_breaks]); # dgram.shape = (N_res, N_res, num_bins)
  return tf.keras.Model(inputs = positions, outputs = dgram);

def pseudo_beta_fn(use_mask = False):
  aatype = tf.keras.Input((None,)); # aatype.shape = (seq_len, N_res)
  all_atom_positions = tf.keras.Input((atom_type_num, 3)); # all_atom_positions.shape = (N_res, atom_type_num, 3)
  if use_mask:
    all_atom_masks = tf.keras.Input((atom_type_num,)); # all_atom_masks.shape = (N_res, atom_type_num)
  is_gly = tf.keras.layers.Lambda(lambda x, g: tf.math.equal(x, g), arguments = {'g': restype_order['G']})(aatype); # is_gly.shape = (seq_len, N_res)
  pseudo_beta = tf.keras.layers.Lambda(lambda x, ca_idx, cb_idx: tf.where(tf.tile(tf.expand_dims(x[0], axis = -1), (1,1,3)),
                                                                          x[1][..., ca_idx, :],
                                                                          x[1][..., cb_idx, :]),
                                       arguments = {'ca_idx': atom_order['CA'], 'cb_idx': atom_order['CB']})([is_gly, all_atom_positions]); # pseudo_beta.shape = (seq_len, N_res, 3)
  if use_mask:
    pseudo_beta_mask = tf.keras.layers.Lambda(lambda x, ca_idx, cb_idx: tf.cast(tf.where(x[0],
                                                                                         x[1][..., ca_idx],
                                                                                         x[1][..., cb_idx]), dtype = tf.float32),
                                              arguments = {'ca_idx': atom_order['CA'], 'cb_idx': atom_order['CB']})([is_gly, all_atom_masks]); # pseudo_beta_mask.shape = (seq_len, N_res)
  return tf.keras.Model(inputs = (aatype, all_atom_positions, all_atom_masks) if use_mask else (aatype, all_atom_positions), outputs = (pseudo_beta, pseudo_beta_mask) if use_mask else pseudo_beta);

def EvoformerIteration(c_m, c_z, is_extra_msa, key_dim = 64, num_head = 4, value_dim = 64, \
    outer_num_channel = 32, outer_first = False, outer_drop_rate = 0., \
    row_num_head = 8, row_drop_rate = 0.15, \
    column_num_head = 8, column_drop_rate = 0., \
    transition_factor = 4, transition_drop_rate = 0., \
    tri_mult_intermediate_channel = 128, tri_mult_drop_rate = 0.25, \
    tri_attn_drop_rate = 0.25):
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  pair_act = tf.keras.Input((None, c_z)); # pair_act.shape = (N_res, N_res, c_z)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  pair_mask = tf.keras.Input((None,)); # pair_mask.shape = (N_res, N_res)
  msa_act_results = msa_act;
  pair_act_results = pair_act;
  msa_mask_results = msa_mask;
  pair_mask_results = pair_mask;
  if outer_first:
    residual = OuterProductMean(c_z, c_m, num_outer_channel = outer_num_channel)([msa_act_results, msa_mask_results]); # residual.shape = (N_res, N_res, c_z)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': outer_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
    pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]); # pair_act_results.shape = (N_res, N_res, c_z)
  residual = MSARowAttentionWithPairBias(c_m, c_z, num_head = row_num_head)([msa_act_results, msa_mask_results, pair_act_results]); # residual.shape = (N_res, N_res, c_m)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': row_drop_rate})(residual); # residual.shape = (N_res, N_res, c_m)
  msa_act_results = tf.keras.layers.Add()([msa_act_results, residual]); # msa_act_results.shape = (N_res, N_res, c_m)
  if not is_extra_msa:
    residual = MSAColumnAttention(c_m, num_head = column_num_head)([msa_act_results, msa_mask_results]); # residual.shape = (N_res, N_res, c_m)
  else:
    residual = MSAColumnGlobalAttention(c_m, num_head = column_num_head)([msa_act_results, msa_mask_results]); # residual.shape = (N_res, N_res, c_m)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (tf.shape(x)[0], 1, tf.shape(x)[2])), arguments = {'r': column_drop_rate})(residual); # residual.shape = (N_res, N_res, c_m)
  msa_act_results = tf.keras.layers.Add()([msa_act_results, residual]); # msa_act_results.shape = (N_res, N_res, c_m)
  residual = Transition(c_m, num_intermediate_factor = transition_factor)([msa_act_results, msa_mask_results]); # residual.shape = (N_res, N_res, c_m)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': transition_drop_rate})(residual); # residual.shape = (N_res, N_res, c_m)
  msa_act_results = tf.keras.layers.Add()([msa_act_results, residual]); # msa_act_results.shape = (N_res, N_res, c_m)
  if not outer_first:
    residual = OuterProductMean(c_z, c_m, num_outer_channel = outer_num_channel)([msa_act_results, msa_mask_results]); # residual.shape = (N_res, N_res, c_z)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': outer_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
    pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]); # pair_act_results.shape = (N_res, N_res, c_z)
  residual = TriangleMultiplication(c_z, intermediate_channel = tri_mult_intermediate_channel, mode = 'outgoing')([pair_act_results, pair_mask_results]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': tri_mult_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]); # pair_act_results.shape = (N_res, N_res, c_z)
  residual = TriangleMultiplication(c_z, intermediate_channel = tri_mult_intermediate_channel, mode = 'incoming')([pair_act_results, pair_mask_results]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': tri_mult_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]); # pair_act_results.shape = (N_res, N_res, c_z)
  residual = TriangleAttention(c_z, num_head = num_head)([pair_act_results, pair_mask_results]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': tri_attn_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]); # pair_act_results.shape = (N_res, N_res, c_z)
  residual = TriangleAttention(c_z, num_head = num_head)([pair_act_results, pair_mask_results]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (tf.shape(x)[0], 1, tf.shape(x)[2])), arguments = {'r': tri_attn_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]); # pair_act_results.shape = (N_res, N_res, c_z)
  residual = Transition(c_z, num_intermediate_factor = transition_factor)([pair_act_results, pair_mask_results]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': transition_drop_rate})(residual); # residual.shape = (N_Res, N_res, c_z)
  pair_act_results = tf.keras.layers.Add()([pair_act_results, residual]); # pair_act_results.shape = (N_res, N_res, c_z)
  return tf.keras.Model(inputs = (msa_act, pair_act, msa_mask, pair_mask), outputs = (msa_act_results, pair_act_results));

def EmbeddingsAndEvoformer(N_seq, N_res, N_template, msa_channel = 256, pair_channel = 128):
  target_feat = tf.keras.Input((None,), batch_size = N_res); # target_feat.shape = (N_res, None)
  msa_feat = tf.keras.Input((N_res, None), batch_size = N_seq); # msa_feat.shape = (N_seq, N_res, None)
  seq_mask = tf.keras.Input((), batch_size = N_res); # seq_mask.shape = (N_res)
  aatype = tf.keras.Input((), batch_size = N_res); # aatype.shape = (N_res)
  prev_pos = tf.keras.Input((atom_type_num, 3), batch_size = N_res); # prev_pos.shape = (N_res, atom_type_num, 3)
  prev_msa_first_row = tf.keras.Input((msa_channel,), batch_size = N_res); # prev_msa_first_row.shape = (N_res, msa_channel)
  prev_pair = tf.keras.Input((N_res, pair_channel), batch_size = N_res); # prev_pair.shape = (N_res, N_res, pair_channel)
  residue_index = tf.keras.Input((), batch_size = N_res); # residue_index.shape = (N_res)
  extra_msa_mask = tf.keras.Input((N_res,), batch_size = N_seq); # extra_msa_mask.shape = (N_seq, N_res)
  template_aatype = tf.keras.Input((N_res, None), batch_size = N_template); # template_aatype.shape = (N_template, N_res, None)
  template_all_atom_positions = tf.keras.Input((N_res, None, None), batch_size = N_template); # template_all_atom_positions.shape = (N_template, N_res, None, None)
  template_all_atom_masks = tf.keras.Input((N_res, None), batch_size = N_template); # template_all_atom_masks.shape = (N_tepmlate, N_res, None)
  preprocess_1d = tf.keras.layers.Dense(msa_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(target_feat); # preprocess_1d.shape = (N_res, msa_channel)
  preprocess_msa = tf.keras.layers.Dense(msa_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(msa_feat); # prreprocess_msa.shape = (N_seq, N_res, msa_channel)
  msa_activations = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0], axis = 0) + x[1])([preprocess_1d, preprocess_msa]); # msa_activations.shape = (N_seq, N_res, msa_channel)
  left_single = tf.keras.layers.Dense(pair_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(target_feat); # left_single.shape = (N_res, pair_channel)
  right_single = tf.keras.layers.Dense(pair_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(target_feat); # right_single.shape = (N_res, pair_channel)
  pair_activations = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0], axis = 1) + tf.expand_dims(x[1], axis = 0))([left_single, right_single]); # pair_activations.shape = (N_res, N_res, pair_channel)
  mask_2d = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 1) * tf.expand_dims(x, axis = 0))(seq_mask); # mask_2d.shape = (N_res, N_res)
  
  return tf.keras.Model(inputs = (target_feat, msa_feat, seq_mask, aatype, prev_pos, prev_msa, prev_pair, residue_index, extra_msa_mask, template_aatype, template_all_atom_positions, template_all_atom_masks,),
                        outputs = ());

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
  results = TemplatePairStack(64)([pair_act, pair_mask]);
  print(results.shape);
  results = Transition(64)([pair_act, pair_mask]);
  print(results.shape);
  results = OuterProductMean(64, 64)([pair_act, pair_mask]);
  print(results.shape);
  positions = np.random.normal(size = (10,3));
  results = dgram_from_positions(3.25, 50.75)(positions);
  print(results.shape);
  all_atom_positions = np.random.normal(size = (10, 37, 3));
  aatype = np.random.normal(size = (50,10));
  all_atom_masks = np.random.randint(low = 0, high = 2, size = (10, 37));
  pseudo_beta = pseudo_beta_fn()([all_atom_positions, aatype]);
  print(pseudo_beta.shape);
  pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(use_mask = True)([all_atom_positions, aatype, all_atom_masks]);
  print(pseudo_beta_mask.shape);
  msa_act = np.random.normal(size = (4, 20, 32));
  pair_act = np.random.normal(size = (20, 20, 64));
  msa_mask = np.random.normal(size = (4, 20));
  pair_mask = np.random.normal(size = (20, 20));
  msa_act, pair_act = EvoformerIteration(32, 64, False)([msa_act, pair_act, msa_mask, pair_mask]);
  print(msa_act.shape, pair_act.shape);
  msa_act, pair_act = EvoformerIteration(32, 64, False, outer_first = True)([msa_act, pair_act, msa_mask, pair_mask]);
  print(msa_act.shape, pair_act.shape);
  msa_act, pair_act = EvoformerIteration(32, 64, True)([msa_act, pair_act, msa_mask, pair_mask]);
  print(msa_act.shape, pair_act.shape);
  msa_act, pair_act = EvoformerIteration(32, 64, True, outer_first = True)([msa_act, pair_act, msa_mask, pair_mask]);
  print(msa_act.shape, pair_act.shape);
