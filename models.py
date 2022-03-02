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

def dgram_from_positions(min_bin, max_bin, num_bins = 39, use_3d = False):
  positions = tf.keras.Input((3,)) if use_3d == False else tf.keras.Input((None, 3)); # positions.shape = (N_res, 3)
  lower_breaks = tf.keras.layers.Lambda(lambda x,l,u,n: tf.linspace(l,u,n), arguments = {'l': min_bin, 'u': max_bin, 'n': num_bins})(positions); # lower_breaks.shape = (num_bins)
  lower_breaks = tf.keras.layers.Lambda(lambda x: tf.math.square(x))(lower_breaks); # lower_breaks.shape = (num_bins,)
  upper_breaks = tf.keras.layers.Lambda(lambda x: tf.concat([x[1:], [1e8]], axis = -1))(lower_breaks); # upper_breaks.shape = (num_bins,)
  dist2 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.square(tf.expand_dims(x, axis = -2) - tf.expand_dims(x, axis = -3)), axis = -1, keepdims = True))(positions); # dist2.shape = (N_res, N_res, 1)
  dgram = tf.keras.layers.Lambda(lambda x: tf.cast(x[0] > x[1], dtype = tf.float32) * tf.cast(x[0] < x[2], dtype = tf.float32))([dist2, lower_breaks, upper_breaks]); # dgram.shape = (N_res, N_res, num_bins)
  return tf.keras.Model(inputs = positions, outputs = dgram);

def pseudo_beta_fn(use_mask = False):
  aatype = tf.keras.Input(()); # aatype.shape = (N_res)
  all_atom_positions = tf.keras.Input((atom_type_num, 3)); # all_atom_positions.shape = (N_res, atom_type_num, 3)
  if use_mask:
    all_atom_masks = tf.keras.Input((atom_type_num,)); # all_atom_masks.shape = (N_res, atom_type_num)
  is_gly = tf.keras.layers.Lambda(lambda x, g: tf.math.equal(x, g), arguments = {'g': restype_order['G']})(aatype); # is_gly.shape = (N_res)
  pseudo_beta = tf.keras.layers.Lambda(lambda x, ca_idx, cb_idx: tf.where(tf.tile(tf.expand_dims(x[0], axis = -1), (1,3)),
                                                                          x[1][..., ca_idx, :],
                                                                          x[1][..., cb_idx, :]),
                                       arguments = {'ca_idx': atom_order['CA'], 'cb_idx': atom_order['CB']})([is_gly, all_atom_positions]); # pseudo_beta.shape = (N_res, N_res, 3)
  if use_mask:
    pseudo_beta_mask = tf.keras.layers.Lambda(lambda x, ca_idx, cb_idx: tf.cast(tf.where(x[0],
                                                                                         x[1][..., ca_idx],
                                                                                         x[1][..., cb_idx]), dtype = tf.float32),
                                              arguments = {'ca_idx': atom_order['CA'], 'cb_idx': atom_order['CB']})([is_gly, all_atom_masks]); # pseudo_beta_mask.shape = (N_res, N_res)
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

def make_canonical_transform():
  n_xyz = tf.keras.Input((3,)); # n_xyz.shape = (batch, 3)
  ca_xyz = tf.keras.Input((3,)); # ca_xyz.shape = (batch, 3)
  c_xyz = tf.keras.Input((3,)); # c_xyz.shape = (batch, 3)
  n_xyz_results = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([n_xyz, ca_xyz]); # n_xyz_results.shape = (batch, 3)
  c_xyz_results = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([c_xyz, ca_xyz]); # c_xyz_results.shape = (batch, 3)
  sin_c1 = tf.keras.layers.Lambda(lambda x: -x[:,1] / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1])))(c_xyz_results); # sin_c1.shape = (batch)
  cos_c1 = tf.keras.layers.Lambda(lambda x: -x[:,0] / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1])))(c_xyz_results); # cos_c1.shape = (batch)
  c1_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([x[1], -x[0], tf.zeros_like(x[0])], axis = -1),
                                                             tf.stack([x[0], x[1], tf.zeros_like(x[0])], axis = -1),
                                                             tf.stack([tf.zeros_like(x[0]), tf.zeros_like(x[0]), tf.ones_like(x[0])], axis = -1)], axis = -2))([sin_c1, cos_c1]); # c1_rot_matrix.shape = (batch, 3, 3)
  sin_c2 = tf.keras.layers.Lambda(lambda x: x[:,2] / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(c_xyz_results); # sin_c2.shape = (batch)
  cos_c2 = tf.keras.layers.Lambda(lambda x: tf.math.sqrt(tf.math.square(x[:,0]) + tf.math.square(x[:,1])) / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(c_xyz_results); # cos_c2.shape = (batch)
  c2_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([x[1], tf.zeros_like(x[0]), x[0]], axis = -1),
                                                             tf.stack([tf.zeros_like(x[0]), tf.ones_like(x[0]), tf.zeros_like(x[0])], axis = -1),
                                                             tf.stack([-x[0], tf.zeros_like(x[0]), x[1]], axis = -1)], axis = -2))([sin_c2, cos_c2]); # c2_rot_matrix.shape = (batch, 3, 3)
  c_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([c2_rot_matrix, c1_rot_matrix]); # c_rot_matrix.shape = (batch, 3, 3)
  n_xyz_results = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.linalg.matmul(x[0], tf.expand_dims(x[1], axis = -1)), axis = -1))([c_rot_matrix, n_xyz_results]); # n_xyz_results.shape = (batch, 3)
  sin_n = tf.keras.layers.Lambda(lambda x: -x[:,2] / tf.math.sqrt(1e-20 + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(n_xyz_results); # sin_n.shape = (batch)
  cos_n = tf.keras.layers.Lambda(lambda x: x[:,1] / tf.math.sqrt(1e-20 + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(n_xyz_results); # cos_n.shape = (batch)
  n_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([tf.ones_like(x[0]), tf.zeros_like(x[0]), tf.zeros_like(x[0])], axis = -1),
                                                            tf.stack([tf.zeros_like(x[0]), x[1], -x[0]], axis = -1),
                                                            tf.stack([tf.zeros_like(x[0]), x[0], x[1]], axis = -1)], axis = -2))([sin_n, cos_n]); # n_rot_matrix.shape = (batch, 3, 3)
  rot_matrix = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([n_rot_matrix, c_rot_matrix]); # rot_matrix.shape = (batch, 3, 3)
  translation = tf.keras.layers.Lambda(lambda x: -x)(ca_xyz); # translation.shape = (batch, 3)
  return tf.keras.Model(inputs = (n_xyz, ca_xyz, c_xyz), outputs = (translation, rot_matrix));

def rot_to_quat(unstack_inputs = False):
  if unstack_inputs:
    rot = tf.keras.Input((atom_type_num, 3, 3)); # rot.shape = (N_template, atom_type_num, 3, 3)
    rot_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (2,3,0,1)))(rot); # rot_results.shape = (3, 3, N_template, atom_type_num)
  else:
    rot = tf.keras.Input((3, None, atom_type_num)); # rot.shape = (3, 3, N_template, atom_type_num)
    rot_results = rot;
  k = tf.keras.layers.Lambda(lambda x: 1/3 * tf.stack([tf.stack([x[0,0] + x[1,1] + x[2,2], x[2,1] - x[1,2], x[0,2] - x[2,0], x[1,0] - x[0,1]], axis = -1),
                                                       tf.stack([x[2,1] - x[1,2], x[0,0] - x[1,1] - x[2,2], x[0,1] + x[1,0], x[0,2] + x[2,0]], axis = -1),
                                                       tf.stack([x[0,2] - x[2,0], x[0,1] + x[1,0], x[1,1] - x[0,0] - x[2,2], x[1,2] + x[2,1]], axis = -1),
                                                       tf.stack([x[1,0] - x[0,1], x[0,2] + x[2,0], x[1,2] + x[2,1], x[2,2] - x[0,0] - x[1,1]], axis = -1)], axis = -2))(rot_results); # x.shape = (N_template, atom_type_num, 4, 4)
  qs = tf.keras.layers.Lambda(lambda x: tf.linalg.eigh(x)[1])(k); # qs.shape = (N_template, atom_type_num, 4, 4)
  # NOTE: return the eigvector of the biggest eigvalue
  qs = tf.keras.layers.Lambda(lambda x: x[...,-1])(qs); # qs.shape = (N_template, atom_type_num, 4)
  return tf.keras.Model(inputs = rot, outputs = qs);

def quat_to_rot():
  normalized_quat = tf.keras.Input(());
  QUAT_TO_ROT = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([[[1,0,0],[0,1,0],[0,0,1]], [[0,0,0],[0,0,-2],[0,2,0]], [[0,0,2],[0,0,0],[-2,0,0]],[[0,-2,0],[2,0,0],[0,0,0]]], axis = 0),
                                                           tf.stack([[[0,0,0],[0,0,0],[0,0,0]], [], [], []], axis = 0),
                                                           tf.stack([[],[],[],[]], axis = 0),
                                                           tf.stack([[],[],[],[]], axis = 0),], axis = 0))(normalized_quat);

def SingleTemplateEmbedding(c_z, min_bin = 3.25, max_bin = 50.75, num_bins = 39):
  query_embedding = tf.keras.Input((None, c_z)); # query_embedding.shape = (N_res, N_res, c_z)
  template_aatype = tf.keras.Input((None,)); # template_aatype.shape = (N_template, N_res,)
  template_all_atom_positions = tf.keras.Input((None, atom_type_num, 3)); # template_all_atom_positions.shape = (N_template, N_res, atom_type_num, 3)
  template_pseudo_beta_mask = tf.keras.Input((None,)); # template_pseudo_beta_mask.shape = (N_template, N_res)
  template_mask = tf.keras.Input(()); # template_mask.shape = (N_template)
  template_pseudo_beta = tf.keras.Input((None, None,)); # template_seudo_beta.shape = (N_template, N_res, None)
  template_pseudo_beta_mask = tf.keras.Input((None,)); # template_seudo_beta_mask.shape = (N_template, N_res)
  template_mask_2d = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.cast(tf.expand_dims(x[0], axis = 1) * tf.expand_dims(x[0], axis = 0), dtype = x[1].dtype), axis = -1))([template_pseudo_beta_mask, query_embedding]); # template_mask_2d.shape = (N_template, N_template, N_res, 1)
  template_dgram = dgram_from_positions(min_bin, max_bin, num_bins, use_3d = True)(template_pseudo_beta); # template_dgram.shape = (N_template, N_res, N_res, num_bins)
  template_dgram = tf.keras.layers.Lambda(lambda x: tf.cast(x[0], dtype = x[1].dtype))([template_dgram, query_embedding]); # template_dgram.shape = (N_template, N_res, N_res, num_bins)
  to_concat = [template_dgram, template_mask_2d];
  aatype = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, 22))(template_aatype); # aatype.shape = (N_template, N_res, 22)
  aatype_tile0 = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x[0], axis = 0), (tf.shape(x[1])[0],1,1)))([aatype, template_aatype]); # aatype_tile0.shape = (N_template, N_template, N_res, 22)
  aatype_tile1 = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x[0], axis = 1), (1,tf.shape(x[1])[0],1)))([aatype, template_aatype]); # aatype_tile1.shape = (N_template, N_template, N_res, 22)
  to_concat.append(aatype_tile0);
  to_concat.append(aatype_tile1);
  n_xyz = tf.keras.layers.Lambda(lambda x, n: tf.reshape(x[:,n], (-1, 3)), arguments = {'n': residue_constants.atom_order['N']})(template_all_atom_positions); # n_xyz.shape = (N_template * atom_type_num, 3)
  ca_xyz = tf.keras.layers.Lambda(lambda x, n: tf.reshape(x[:,n], (-1, 3)), arguments = {'n': residue_constants.atom_order['CA']})(template_all_atom_positions); # ca_xyz.shape = (N_template * atom_type_num, 3)
  c_xyz = tf.keras.layers.Lambda(lambda x, n: tf.reshape(x[:,n], (-1, 3)), arguments = {'n': residue_constants.atom_order['C']})(template_all_atom_positions); # c_xyz.shape = (N_template * atom_type_num, 3)
  translation, rot_matrix = make_canonical_transform()([n_xyz, ca_xyz, c_xyz]); # translation.shape = (N_template * atom_type_num, 3) rot_matrix.shape = (N_template * atom_type_num, 3, 3)
  # INFO: get inverse transformation (rotation, translation)
  trans = tf.keras.layers.Lambda(lambda x, n: tf.reshape(-x, (-1, n, 3)), arguments = {'n': atom_type_num})(translation); # trans.shape = (N_template, atom_type_num, 3)
  rot = tf.keras.layers.Lambda(lambda x, n: tf.reshape(tf.transpose(x, (0, 2, 1)), (-1, n, 3, 3)), arguments = {'n': atom_type_num})(rot_matrix); # rot.shape = (N_template, atom_type_num, 3, 3)
  quaternion = rot_to_quat(unstack_inputs = True)(rot); # quaternion.shape = (N_template, atom_type_num, 4)

def TemplateEmbedding(c_z):
  query_embedding = tf.keras.Input((None, c_z)); # query_embedding.shape = (N_res, N_res, c_z)
  template_mask = tf.keras.Input(()); # template_mask.shape = (N_template)
  mask_2d = tf.keras.Input((None,)); # mask_2d.shape = (N_res, N_res)
  template_mask_results = tf.keras.layers.Lambda(lambda x: tf.cast(x[0], dtype = x[1].dtype))([template_mask, query_embedding]); # template_mask_results.shape = (N_template)

def EmbeddingsAndEvoformer(c_m = 22, c_z = 25, msa_channel = 256, pair_channel = 128, recycle_pos = True, prev_pos_min_bin = 3.25, prev_pos_max_bin = 20.75, prev_pos_num_bins = 15, recycle_features = True, max_relative_feature = 32, template_enabled = False, extra_msa_channel = 64, extra_msa_stack_num_block = 4, evoformer_num_block = 48, seq_channel = 384):
  target_feat = tf.keras.Input((c_m,)); # target_feat.shape = (N_res, c_m)
  msa_feat = tf.keras.Input((None, c_z)); # msa_feat.shape = (N_seq, N_res, c_z)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  seq_mask = tf.keras.Input(()); # seq_mask.shape = (N_res)
  aatype = tf.keras.Input(()); # aatype.shape = (N_res)
  prev_pos = tf.keras.Input((atom_type_num, 3)); # prev_pos.shape = (N_res, atom_type_num, 3)
  prev_msa_first_row = tf.keras.Input((msa_channel,)); # prev_msa_first_row.shape = (N_res, msa_channel)
  prev_pair = tf.keras.Input((None, pair_channel)); # prev_pair.shape = (N_res, N_res, pair_channel)
  residue_index = tf.keras.Input((), dtype = tf.int32); # residue_index.shape = (N_res)
  extra_msa = tf.keras.Input((None,), dtype = tf.int32); # extra_msa.shape = (N_seq, N_res)
  extra_msa_mask = tf.keras.Input((None,)); # extra_msa_mask.shape = (N_seq, N_res)
  extra_has_deletion = tf.keras.Input((None,)); # extra_has_deletion.shape = (N_seq, N_res)
  extra_deletion_value = tf.keras.Input((None,)); # extra_deletion_value.shape = (N_seq, N_res)

  preprocess_1d = tf.keras.layers.Dense(msa_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(target_feat); # preprocess_1d.shape = (N_res, msa_channel)
  preprocess_msa = tf.keras.layers.Dense(msa_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(msa_feat); # prreprocess_msa.shape = (N_seq, N_res, msa_channel)
  msa_activations = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0], axis = 0) + x[1])([preprocess_1d, preprocess_msa]); # msa_activations.shape = (N_seq, N_res, msa_channel)
  left_single = tf.keras.layers.Dense(pair_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(target_feat); # left_single.shape = (N_res, pair_channel)
  right_single = tf.keras.layers.Dense(pair_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(target_feat); # right_single.shape = (N_res, pair_channel)
  pair_activations = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0], axis = 1) + tf.expand_dims(x[1], axis = 0))([left_single, right_single]); # pair_activations.shape = (N_res, N_res, pair_channel)
  mask_2d = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 1) * tf.expand_dims(x, axis = 0))(seq_mask); # mask_2d.shape = (N_res, N_res)
  if recycle_pos:
    prev_pseudo_beta = pseudo_beta_fn(use_mask = False)([aatype, prev_pos]); # prev_pseudo_beta.shape = (N_res, N_res, 3)
    dgram = dgram_from_positions(prev_pos_min_bin, prev_pos_max_bin, prev_pos_num_bins)(prev_pseudo_beta); # dgram.shape = (N_res, N_res, num_bins)
    dgram = tf.keras.layers.Dense(pair_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(dgram); # dgram.shape = (N_res, N_res, pair_channel)
    pair_activations = tf.keras.layers.Add()([pair_activations, dgram]); # pair_activations.shape = (N_res, N_res, pair_channel)
  if recycle_features:
    prev_msa_first_row_results = tf.keras.layers.LayerNormalization()(prev_msa_first_row); # prev_msa_first_row_results.shape = (N_res, msa_channel)
    msa_activations = tf.keras.layers.Lambda(lambda x: tf.concat([tf.expand_dims(x[1], axis = 0), tf.zeros((tf.shape(x[0])[0] - 1, tf.shape(x[0])[1], tf.shape(x[0])[2]))], axis = 0) + x[0])([msa_activations, prev_msa_first_row_results]); # msa_activations.shape = (N_seq, N_res, msa_channel)
    prev_pair_results = tf.keras.layers.LayerNormalization()(prev_pair); # prev_pair_results.shape = (N_res, N_res, pair_channel)
    pair_activations = tf.keras.layers.Add()([pair_activations, prev_pair_results]); # pair_activations.shape = (N_res, N_res, pair_channel)
  offset = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 1) - tf.expand_dims(x, axis = 0))(residue_index); # offset.shape = (N_res, N_res)
  rel_pos = tf.keras.layers.Lambda(lambda x, r: tf.one_hot(tf.clip_by_value(x + r, 0, 2 * r), 2 * r + 1), arguments = {'r': max_relative_feature})(offset); # rel_pos.shape = (N_res, N_res, 2 * max_relative_feature + 1)
  rel_pos = tf.keras.layers.Dense(pair_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(rel_pos); # rel_pos.shape = (N_res, N_res, pair_channel)
  pair_activations = tf.keras.layers.Add()([pair_activations, rel_pos]); # pair_activations.shape = (N_res, N_res, pair_channel)
  if template_enabled:
    # TODO: will implement in the future
    pass;
  msa_1hot = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, 23))(extra_msa); # msa_1hot.shape = (N_seq, N_res, 23)
  extra_msa_feat = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], axis = -1), tf.expand_dims(x[2], axis = -1)], axis = -1))([msa_1hot, extra_has_deletion, extra_deletion_value]); # extra_msa_feat.shape = (N_seq, N_res, 25)
  extra_msa_activations = tf.keras.layers.Dense(extra_msa_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(extra_msa_feat); # extra_msa_activations.shape = (N_seq, N_res, extra_msa_channel)
  # Embed extra MSA features
  msa = extra_msa_activations;
  pair = pair_activations;
  for i in range(extra_msa_stack_num_block):
    # msa.shape = (N_seq, N_res, extra_msa_channel)
    # pair.shape = (N_seq, N_res, pair_channel)
    msa, pair = EvoformerIteration(extra_msa_channel, pair_channel, is_extra_msa = True)([msa, pair, extra_msa_mask, mask_2d]);
  pair_activations = pair; # pair_activations.shape = (N_seq, N_res, pair_channel)
  if template_enabled:
    # TODO: will implement in the future
    pass;
  # Embed MSA features
  msa_act = msa_activations;
  pair_act = pair_activations;
  for i in range(evoformer_num_block):
    # msa_act.shape = (N_seq, N_res, msa_channel)
    # pair_act.shape = (N_seq, N_res, pair_channel)
    msa_act, pair_act = EvoformerIteration(msa_channel, pair_channel, is_extra_msa = False)([msa_act, pair_act, msa_mask, mask_2d]);
  msa_activations = msa; # msa_activations.shape = (N_seq, N_res, msa_channel)
  pair_activations = pair; # pair_activations.shape = (N_seq, N_res, pair_channel)
  
  single_msa_activations = tf.keras.layers.Lambda(lambda x: x[0])(msa_activations); # single_msa_activations.shape = (N_res, msa_channel)
  single_activations = tf.keras.layers.Dense(seq_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(single_msa_activations); # single_activations.shape = (N_res, seq_channel)
  msa = tf.keras.layers.Lambda(lambda x: x[0][:tf.shape(x[1])[0],:,:])([msa_activations, msa_feat]); # msa.shape = (N_seq, N_res, msa_channel)
  return tf.keras.Model(inputs = (target_feat, msa_feat, msa_mask, seq_mask, aatype, prev_pos, prev_msa_first_row, prev_pair, residue_index, extra_msa, extra_msa_mask, extra_has_deletion, extra_deletion_value,),
                        outputs = (single_activations, pair_activations, msa, single_msa_activations));

def AlphaFoldIteration():
  seq_length = tf.keras.Input();
  

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
  print('dgram2d.shape = ', results.shape);
  positions = np.random.normal(size = (4,10,3));
  results = dgram_from_positions(3.25, 50.75, use_3d = True)(positions);
  print('dgram3d.shape = ', results.shape);
  all_atom_positions = np.random.normal(size = (10, 37, 3));
  aatype = np.random.normal(size = (10,));
  all_atom_masks = np.random.randint(low = 0, high = 2, size = (10, 37));
  pseudo_beta = pseudo_beta_fn()([aatype, all_atom_positions]);
  print(pseudo_beta.shape);
  pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(use_mask = True)([aatype, all_atom_positions, all_atom_masks]);
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
  '''
  target_feat = np.random.normal(size = (20, 22));
  msa_feat = np.random.normal(size = (4, 20, 25));
  msa_mask = np.random.normal(size = (4, 20));
  seq_mask = np.random.normal(size = (20,));
  aatype = np.random.normal(size = (20,));
  prev_pos = np.random.normal(size = (20, 37, 3));
  prev_msa_first_row = np.random.normal(size = (20, 256));
  prev_pair = np.random.normal(size = (20, 20, 128));
  residue_index = np.random.randint(low = 0, high = 10, size = (20,));
  extra_msa = np.random.randint(low = 0, high = 10, size = (4, 20));
  extra_msa_mask = np.random.normal(size = (4, 20));
  extra_has_deletion = np.random.normal(size = (4, 20));
  extra_deletion_value = np.random.normal(size = (4, 20));
  single_activations, pair_activations, msa_activations, single_msa_activations = EmbeddingsAndEvoformer()([target_feat, msa_feat, msa_mask, seq_mask, aatype, prev_pos, prev_msa_first_row, prev_pair, residue_index, extra_msa, extra_msa_mask, extra_has_deletion, extra_deletion_value]);
  print(single_activations.shape);
  print(pair_activations.shape);
  print(msa_activations.shape);
  '''
  n_xyz = np.random.normal(size = (10,3));
  ca_xyz = np.random.normal(size = (10,3));
  c_xyz = np.random.normal(size = (10,3));
  translation, rot_matrix = make_canonical_transform()([n_xyz, ca_xyz, c_xyz]);
  print(translation.shape, rot_matrix.shape);
  rot = np.random.normal(size = (4, atom_type_num, 3, 3));
  quat = rot_to_quat(True)(rot);
  print(quat.shape);
  rot = np.random.normal(size = (3, 3, 4, atom_type_num));
  quat = rot_to_quat(False)(rot);
  print(quat.shape);
