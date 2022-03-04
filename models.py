#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from residue_constants import restype_order, atom_order, atom_type_num;

def TemplatePairStack(c_t, num_head = 4, num_intermediate_channel = 64, num_block = 2, rate = 0.25, **kwargs):
  pair_act = tf.keras.Input((None, c_t)); # pair_act.shape = (N_res, N_res, c_t)
  pair_mask = tf.keras.Input((None, )); # pair_mask.shape = (N_res, N_res)
  inputs = (pair_act, pair_mask);
  for i in range(num_block):
    # triangle_attention_starting_node
    residual = TriangleAttention(c_t, num_head = num_head, orientation = 'per_row', name = 'block%d/triangle_attention_starting_node' % i)([pair_act, pair_mask]); # pair_act.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': rate})(residual); # pair_act.shape = (N_res, N_res, c_t)
    pair_act = tf.keras.layers.Add()([pair_act, residual]);
    # triangle_attention_ending_node
    residual = TriangleAttention(c_t, num_head = num_head, orientation = 'per_column', name = 'block%d/triangle_attention_ending_node' % i)([pair_act, pair_mask]); # pair_act.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (tf.shape(x)[0], 1, tf.shape(x)[2])), arguments = {'r': rate})(residual); # pair_act.shape = (N_res, N_res, c_t)
    pair_act = tf.keras.layers.Add()([pair_act, residual]);
    # triangle_multiplication_outgoing
    residual = TriangleMultiplication(c_t, intermediate_channel = num_intermediate_channel, mode = 'outgoing', name = 'block%d/triangle_multiplication_outgoing' % i)([pair_act, pair_mask]); # residual.shape = (N_res, N_res, c_t)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': rate})(residual); # residual.shape = (N_res, N_res, c_t)
    pair_act = tf.keras.layers.Add()([pair_act, residual]);
    # triangle_multiplication_incoming
    residual = TriangleMultiplication(c_t, intermediate_channel = num_intermediate_channel, mode = 'incoming', name = 'block%d/triangle_multiplication_incoming' % i)([pair_act, pair_mask]); # residual.shape = (N_res, N_res, act)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': rate})(residual); # residual.shape = (N_res, N_res, c_t)
    pair_act = tf.keras.layers.Add()([pair_act, residual]);
  return tf.keras.Model(inputs = inputs, outputs = pair_act, **kwargs);

def Transition(c_t, num_intermediate_factor = 4):
  act = tf.keras.Input((None, c_t)); # act.shape = (batch, N_res, c_t)
  mask = tf.keras.Input((None,)); # mask.shape = (batch, N_res)
  inputs = (act, mask);
  mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(mask); # mask.shape = (batch, N_res, 1)
  act = tf.keras.layers.LayerNormalization()(act); # act.shape = (batch, N_res, c_t)
  act = tf.keras.layers.Dense(c_t * num_intermediate_factor, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = np.sqrt(2)))(act); # act.shape = (batch, N_res, 4*c_t)
  act = tf.keras.layers.Dense(c_t, kernel_initializer = tf.keras.initializers.Zeros())(act); # act.shape = (batch, N_res, c_t)
  return tf.keras.Model(inputs = inputs, outputs = act);

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
  inputs = (msa_act, msa_mask, pair_act);
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask); # bias.shape = (N_seq, num_head = 1, N_queries = 1, N_res)
  msa_act = tf.keras.layers.LayerNormalization()(msa_act);
  pair_act = tf.keras.layers.LayerNormalization()(pair_act); # pair_act.shape = (N_res, N_res, c_z)
  nonbatched_bias = tf.keras.layers.Dense(num_head, use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1./np.sqrt(c_z)))(pair_act); # nonbatched_bias.shape = (N_res. N_res, num_head)
  nonbatched_bias = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (2, 0, 1)))(nonbatched_bias); # nonbatched_bias.shape = (num_head, N_res, N_res)
  msa_act = Attention(c_m, key_dim = c_m, num_head = num_head, value_dim = c_m, use_nonbatched_bias = True)([msa_act, msa_act, bias, nonbatched_bias]); # msa_act.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = inputs, outputs = msa_act, **kwargs);

def MSAColumnAttention(c_m, num_head = 4, **kwargs):
  # NOTE: multi head self attention: query is msa_act.T, key is msa_act.T, value is msa_act.T.
  # NOTE: differences
  # 1) use msa_mask to control bias, bias's shape is N_res(batch) x num_head(1) x N_queries(1) x N_seq.
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  inputs = (msa_act, msa_mask);
  msa_act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act); # msa_act.shape = (N_res, N_seq, c_m)
  msa_mask = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(msa_mask); # msa_mask.shape = (N_res, N_seq)
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask); # bias.shape = (N_res, 1, 1, N_seq)
  msa_act = tf.keras.layers.LayerNormalization()(msa_act); # msa_act.shape = (N_res, N_seq, c_m)
  msa_act = Attention(c_m, key_dim = c_m, num_head = num_head, value_dim = c_m, use_nonbatched_bias = False)([msa_act, msa_act, bias]); # msa_act.shape = (N_res, N_seq, c_m)
  msa_act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act); # msa_act.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = inputs, outputs = msa_act, **kwargs);

def MSAColumnGlobalAttention(c_m, num_head = 4, **kwargs):
  # NOTE: multi head self attention: query is msa_act.T, key is msa_act.T, value is msa_act.T.
  # NOTE: differences
  # 1) use msa_mask to control q_mask which controls bias in global attention.
  msa_act = tf.keras.Input((None, c_m)); # msa_act.shape = (N_seq, N_res, c_m)
  msa_mask = tf.keras.Input((None,)); # msa_mask.shape = (N_seq, N_res)
  inputs = (msa_act, msa_mask);
  msa_act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act); # msa_act.shape = (N_res, N_seq, c_m)
  msa_mask = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(msa_mask); # msa_mask.shape = (N_res, N_seq)
  #bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(msa_mask); # bias.shape = (N_res, 1, 1, N_seq)
  msa_act = tf.keras.layers.LayerNormalization()(msa_act); # msa_act.shape = (N_res, N_seq, c_m)
  msa_mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(msa_mask); # msa_mask.shape = (N_res, N_seq, 1)
  msa_act = GlobalAttention(c_m, key_dim = c_m, num_head = num_head, value_dim = c_m)([msa_act, msa_act, msa_mask]); # msa_act.shape = (N_res, N_seq, c_m)
  msa_act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(msa_act); # msa_act.shape = (N_seq, N_res, c_m)
  return tf.keras.Model(inputs = inputs, outputs = msa_act, **kwargs);

def TriangleAttention(c_z, num_head = 4, orientation = 'per_column', **kwargs):
  # NOTE: multi head self attention: query is pair_act, key is pair_act, value is pair_act.
  # NOTE: difference:
  # 1) use pair_mask to control bias.
  # 2) use pair_act to control nonbatched_bias.
  assert orientation in ['per_column', 'per_row'];
  pair_act = tf.keras.Input((None, c_z)); # pair_act.shape = (N_res, N_res, c_z)
  pair_mask = tf.keras.Input((None,)); # pair_mask.shape = (N_res, N_res)
  inputs = (pair_act, pair_mask);
  if orientation == 'per_column':
    pair_act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(pair_act); # pair_act.shape = (N_res, N_res, c_z)
    pair_mask = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(pair_mask); # pair_mask.shape = (N_res, N_res)
  bias = tf.keras.layers.Lambda(lambda x: tf.reshape(1e9 * (x - 1.), (tf.shape(x)[0], 1, 1, tf.shape(x)[1])))(pair_mask); # bias.shape = (N_seq, 1, 1, N_res) if per_row else (N_res, 1, 1, N_seq)
  pair_act = tf.keras.layers.LayerNormalization()(pair_act); # pair_act.shape = (N_res, N_res, c_z)
  nonbatched_bias = tf.keras.layers.Dense(num_head, use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 1./np.sqrt(c_z)))(pair_act); # nonbatched_bias.shape = (N_res, N_res, num_head)
  nonbatched_bias = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (2, 0, 1)))(nonbatched_bias); # nonbatched_bias.shape = (num_head, N_res, N_res)
  pair_act = Attention(c_z, key_dim = c_z, num_head = num_head, value_dim = c_z, use_nonbatched_bias = True)([pair_act, pair_act, bias, nonbatched_bias]); # pair_act.shape = (N_res, N_res, c_z)
  if orientation == 'per_column':
    pair_act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(pair_act); # pair_act.shape = (N_res, N_res, c_z)
  return tf.keras.Model(inputs = inputs, outputs = pair_act, **kwargs);

def TriangleMultiplication(c_z, intermediate_channel = 64, mode = 'outgoing', **kwargs):
  assert mode in ['outgoing', 'incoming'];
  act = tf.keras.Input((None, c_z)); # act.shape = (N_res, N_res, c_z)
  mask = tf.keras.Input((None,)); # mask.shape = (N_res, N_res)
  inputs = (act, mask);
  mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(mask); # mask.shape = (N_res, N_res, 1)
  act = tf.keras.layers.LayerNormalization()(act); # act.shape = (N_res, N_res, c_z)
  input_act = act;
  # left projection
  left_projection = tf.keras.layers.Dense(intermediate_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act); # left_projection.shape = (N_res, N_res, intermediate_channel)
  left_proj_act = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask, left_projection]); # left_proj_act.shape = (N_res, N_res, intermediate_channel)
  # right projection
  right_projection = tf.keras.layers.Dense(intermediate_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act); # right_projection.shape = (N_res, N_res, intermediate_channel)
  right_proj_act = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask, right_projection]); # right_proj_act.shape = (N_res, N_res, intermediate_channel)
  # left gate
  left_gate_values = tf.keras.layers.Dense(intermediate_channel, activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Constant(1.))(act); # left_gate_values.shape = (N_res, N_res, intermediate_channel)
  right_gate_values = tf.keras.layers.Dense(intermediate_channel, activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Constant(1.))(act); # right_gate_values.shape = (N_res, N_res, intermediate_channel)
  # gate projection
  left_proj_act = tf.keras.layers.Multiply()([left_proj_act, left_gate_values]); # left_proj_act.shape = (N_res, N_res, intermediate_channel)
  right_proj_act = tf.keras.layers.Multiply()([right_proj_act, right_gate_values]); # right_proj_act.shape = (N_res, N_res, intermediate_channel)
  # apply equation
  if mode == 'outgoing':
    act = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(tf.transpose(x[0], (2,0,1)), tf.transpose(x[1], (2,0,1)), transpose_b = True), (1,2,0)))([left_proj_act, right_proj_act]); # act.shape = (N_res, N_res, intermediate_channel)
  else:
    act = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(tf.transpose(x[0], (2,1,0)), tf.transpose(x[1], (2,1,0)), transpose_b = True), (2,1,0)))([left_proj_act, right_proj_act]); # act.shape = (N_res, N_res, intermediate_channel)
  act = tf.keras.layers.LayerNormalization()(act); # act.shape = (N_res, N_res, intermediate_channel)
  act = tf.keras.layers.Dense(c_z, kernel_initializer = tf.keras.initializers.Zeros())(act); # act.shape = (N_res, N_res, c_z)
  gate_values = tf.keras.layers.Dense(c_z, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Constant(1.))(input_act);
  act = tf.keras.layers.Multiply()([act, gate_values]); # act.shape = (N_res, N_res, c_z)
  return tf.keras.Model(inputs = inputs, outputs = act, **kwargs);

def MaskedMsaHead(c_m, num_output = 23, **kwargs):
  msa = tf.keras.Input((None, c_m)); # msa.shape = (N_seq, N_seq, c_m)
  logits = tf.keras.layers.Dense(num_output, kernel_initializer = tf.keras.initializers.Zeros())(msa);
  return tf.keras.Model(inputs = msa, outputs = logits, **kwargs);

class AttentionQK(tf.keras.layers.Layer):
  def __init__(self, num_head = 4, num_point_qk = 4, **kwargs):
    self.num_head = 4;
    self.num_point_qk = num_point_qk;
    super(WeightedOutput, self).__init__(**kwargs);
  def build(self, input_shape):
    self.point_weights = self.add_weights(shape = (self.num_head,), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Constant(np.log(np.exp(1.) - 1.)), name = 'trainable_point_weights');
  def call(self, inputs):
    # inputs.shape = (num_head, N_res, N_res, num_point_qk)
    point_variance = tf.maximum(self.num_point_qk, 1) * 9. / 2; # point_variance.shape = ()
    point_weights = tf.sqrt(1. / (3 * point_variance)); # point_weights.shape = ()
    point_weights = point_weights * tf.expand_dims(self.point_weights, axis = 1); # point_weights.shape = (num_head, 1)
    attn_qk_point = -0.5 * tf.math.reduce_sum(tf.reshape(point_weights, (self.num_head, 1, 1, 1)) * inputs, axis = -1); # attn_qk_point.shape = (num_head, N_res, N_res)
    return attn_qk_point;
  def get_config(self):
    config = super(WeightedOutput, self).get_config();
    config['num_head'] = self.num_head;
    config['num_point_qk'] = self.num_point_qk;
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

def InvariantPointAttention(
    pair_channel = 128, num_channel = 384,
    num_head = 12, num_scalar_qk = 16, num_scalar_v = 16, num_point_qk = 4, num_point_v = 8):
  inputs_1d = tf.keras.Input((num_channel,)); # inputs_1d.shape = (N_res, num_channel)
  inputs_2d = tf.keras.Input((None, pair_channel)); # inputs_2d.shape = (N_res, N_res, pair_channel)
  mask = tf.keras.Input((1,)); # mask.shape = (N_res, 1)
  rotation = tf.keras.Input((3, 3)); # rotation.shape = (N_res, 3,3)
  translation = tf.keras.Input((3,)); # translation.shape = (N_res, 3)
  q_scalar = tf.keras.layers.Dense(num_head * num_scalar_qk, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(inputs_1d); # q_scalar.shape = (N_res, num_head * num_scalar_qk)
  q_scalar = tf.keras.layers.Reshape((num_head, num_scalar_qk))(q_scalar); # q_scalar.shape = (N_res, num_head, num_scalar_qk)
  kv_scalar = tf.keras.layers.Dense(num_head * (num_scalar_v + num_scalar_qk), kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(inputs_1d); # kv_scalar.shape = (N_res, num_head * (num_scalar_v + num_scalar_qk))
  kv_scalar = tf.keras.layers.Reshape((num_head, num_scalar_v + num_scalar_qk))(kv_scalar); # kv_scalar.shape = (N_res, num_head, num_scalar_v + num_scalar_qk)
  k_scalar, v_scalar = tf.keras.layers.Lambda(lambda x, n: tf.split(x, [n,], axis = -1), arguments = {'n': num_scalar_qk})(kv_scalar); # k_scalar.shape = (N_res, num_head, num_scalar_qk), v_scalar.shape = (N_res, num_head, num_scalar_v)
  q_point_local = tf.keras.layers.Dense(num_head * 3 * num_point_qk, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(inputs_1d); # q_point_local.shape = (N_res, num_head * 3 * num_point_qk)
  q_point_local = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 3)))(q_point_local); # q_point_local.shape = (num_head * num_point_qk, 3)
  q_point_global = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[1], x[0], transpose_b = True) + tf.expand_dims(x[2], axis = -1))([q_point_local, rotation, translation]); # q_point_global.shape = (N_res, 3, num_head * num_point_qk)
  q_point_global = tf.keras.layers.Lambda(lambda x, h, p: tf.reshape(tf.transpose(x, (1,0,2)), (3, tf.shape(x)[0], h, p)), arguments = {'h': num_head, 'p': num_point_qk})(q_point_global); # q_point.global.shape = (3, N_res, num_head, num_point_qk)
  kv_point_local = tf.keras.layers.Dense(num_head * 3 * (num_point_qk + num_point_v), kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(inputs_1d); # kv_point_local.shape = (N_res, num_head * 3 * (num_point_qk + num_point_v))
  kv_point_local = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 3)))(kv_point_local); # kv_point_local.shape = (num_head * (num_point_qk + num_point_v), 3)
  kv_point_global = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[1], x[0], transpose_b = True) + tf.expand_dims(x[2], axis = 1))([kv_point_local, rotation, translation]); # kv_point_globa.shape = (N_res, 3, num_head * (num_point_qk + num_point_v))
  kv_point_global = tf.keras.layers.Lambda(lambda x, h, p, v: tf.reshape(tf.transpose(x, (1,0,2)), (3, tf.shape(x)[0], h, p + v)), arguments = {'h': num_head, 'p': num_point_qk, 'v': num_point_v}); # kv_point_global.shape = (3, N_res, num_head, num_point_qk + num_point_v)
  k_point, v_point = tf.keras.layers.Lambda(lambda x, n: tf.split(x, [n,], axis = -1), arguments = {'n': num_point_qk})(kv_point_global); # k_point.shape = (3, N_res, num_head, num_point_qk), v_point.shape = (3, N_res, num_head, num_point_v)
  v_point = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(v_point); # v_point.shape = (3, num_head, N_res, num_point_v)
  q_point = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(q_point_global); # q_point.shape = (3, num_head, N_res, num_point_qk)
  k_point = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(k_point); # k_point.shape = (3, num_head, N_res, num_point_qk)
  dist2 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.square(tf.expand_dims(x[0], axis = 3) - tf.expand_dims(x[1], axis = 2)), axis = 0))([q_point, k_point]); # dist2.shape = (num_head, N_res, N_res, num_point_qk)
  attn_qk_point = AttentionQK(num_head, num_point_qk)(dist2); # attn_qk_point.shape = (num_head, N_res, N_res)
  v = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(v_scalar); # v.shape = (num_head, N_res, num_scalar_v)
  q = tf.keras.layers.Lambda(lambda x, w: tf.transpose(w*x, (1,0,2)), arguments = {'w': np.sqrt(1./ (3 * max(num_scalar_qk, 1)))})(q_scalar); # q.shape = (num_head, N_res, num_scalar_qk)
  k = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(k_scalar); # k.shape = (num_head, N_res, num_scalar_qk)
  attn_qk_scalar = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(q, k, transpose_b = True))([q,k]); # attn_qk_scalar.shape = (num_head, N_res, N_res)
  attn_logits = tf.keras.layers.Add()([attn_qk_scalar, attn_qk_point]); # attn_logits.shape = (num_head, N_res, N_res)
  attention_2d = tf.keras.layers.Dense(num_head, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(inputs_2d); # attention_2d.shape = (N_res, N_res, num_head)
  attention_2d = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (2,0,1)))(attention_2d); # attention_2d.shape = (num_head, N_res, N_res)
  attn_logits = tf.keras.layers.Add()([attn_logits, attention_2d]); # attn_logits.shape = (num_head, N_res, N_res)
  mask_2d = tf.keras.layers.Lambda(lambda x: x * tf.transpose(x, (1,0)))(mask); # mask_2d.shape = (N_res, N_res)
  attn_logits = tf.keras.layers.Lambda(lambda x: x[0] - 1e5 * (1. - x[1]))([attn_logits, mask_2d]); # attn_logits.shape = (num_head, N_res, N_res)
  attn = tf.keras.layers.Softmax()(attn_logits); # attn.shape = (num_head, N_res, N_res)
  result_scalar = tf.keras.layers.Lambda()();

def FoldIteration(num_channel = 384):
  act = tf.keras.Input((num_channel,)); # act.shape = (N_res, num_channel)
  affine = tf.keras.Input((7,)); # affine.shape = (N_res, 7)
  normalized_quat, translation = tf.keras.layers.Lambda(lambda x: tf.split(x, [4,], axis = -1))(affine); # quaternion.shape = (N_res, 4), translation.shape = (N_res, 3)
  rotation = quat_to_rot()(normalized_quat); # rotation.shape = (N_res,3,3)
  # TODO

def StructureModule(seq_channel = 384, pair_channel = 128, num_channel = 384):
  seq_mask = tf.keras.Input(()); # seq_mask.shape = (N_res)
  single = tf.keras.Input((seq_channel,)); # single.shape = (N_res, seq_channel)
  pair = tf.keras.Input((None, pair_channel)); # pair.shape = (N_res, N_res, pair_channel)
  sequence_mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(seq_mask);
  act = tf.keras.layers.LayerNormalization()(single); # act.shape = (N_Res, seq_channel)
  initial_act = act;
  act = tf.keras.layers.Dense(num_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'), bias_initializer = tf.keras.initializers.Constant(0.))(act); # act.shape = (N_res, num_channel)
  quaternion = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims([1.,0.,0.,0.], axis = 0), (tf.shape(x)[0], 1)))(sequence_mask); # quaternion.shape = (N_res, 4)
  translation = tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 3)))(sequence_mask); # translation.shape = (N_res, 3)
  normalized_quat = tf.keras.layers.Lambda(lambda x: x / tf.norm(x, axis = -1, keepdims = True))(quaternion); # quaternion.shape = (N_res, 4)
  affine = tf.keras.layers.Concatenate(axis = -1)([normalized_quat, translation]); # affine.shape = (N_res, 7);
  act_2d = tf.keras.layers.LayerNormalization()(pair); # act_2d.shape = (N_res, N_res, pair_channel)
  

def PredictedLDDTHead(c_s, num_channels = 128, num_bins = 50, **kwargs):
  act = tf.keras.Input((c_s,)); # act.shape = (N_res, c_s)
  inputs = (act,);
  act = tf.keras.layers.LayerNormalization()(act); # act.shape = (N_res, c_s)
  act = tf.keras.layers.Dense(num_channels, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = np.sqrt(2)))(act);
  act = tf.keras.layers.Dense(num_channels, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = np.sqrt(2)))(act);
  logits = tf.keras.layers.Dense(num_bins, kernel_initializer = tf.keras.initializers.Zeros())(act);
  return tf.keras.Model(inputs = inputs, outputs = logits, **kwargs);

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
  inputs = (act, mask);
  mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(mask); # mask.shape = (N_seq, N_res, 1)
  act = tf.keras.layers.LayerNormalization()(act); # act.shape = (N_seq, N_res, c_m)
  left_act = tf.keras.layers.Dense(num_outer_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act); # left_act.shape = (N_seq, N_res, num_outer_channel)
  left_act = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask, left_act]); # left_act.shape = (N_seq, N_res, num_outer_channel)
  right_act = tf.keras.layers.Dense(num_outer_channel, kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'))(act); # right_act.shape = (N_seq, N_res, num_outer_channel)
  right_act = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([mask, right_act]); # right_act.shape = (N_seq, N_res, num_outer_channel)
  left_act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(left_act); # left_act.shape = (N_seq, num_outer_channel, N_res)
  act = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.reshape(tf.linalg.matmul(tf.reshape(x[0], (tf.shape(x[0])[0], -1)), tf.reshape(x[1], (tf.shape(x[1])[0], -1)), transpose_a = True), (tf.shape(x[0])[1], tf.shape(x[0])[2], tf.shape(x[1])[1], tf.shape(x[1])[2])), (2,1,0,3)))([left_act, right_act]); # act.shape = (N_res, N_res, num_outer_channel, num_outer_channel)
  act_reshape = tf.keras.layers.Lambda(lambda x, n: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], n**2)), arguments = {'n': num_outer_channel})(act); # act.shape = (N_res, N_res, num_outer_channel**2)
  act = tf.keras.layers.Dense(num_output_channel, kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(act_reshape); # act.shape = (N_res, N_res, num_output_channel)
  act = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0,2)))(act); # act.shape = (N_res, N_res, num_output_channel)
  norm = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(tf.transpose(x, (2,1,0)), tf.transpose(x,(2,1,0)), transpose_b = True), (1,2,0)))(mask); # norm.shape = (N_res, N_res, 1)
  act = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + 1e-3))([act, norm]); # act.shape = (N_res, N_res, num_output_channel)
  return tf.keras.Model(inputs = inputs, outputs = act);

def dgram_from_positions(min_bin, max_bin, num_bins = 39):
  positions = tf.keras.Input((3,)); # positions.shape = (N_res, 3)
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
  inputs = (msa_act, pair_act, msa_mask, pair_mask);
  if outer_first:
    residual = OuterProductMean(c_z, c_m, num_outer_channel = outer_num_channel)([msa_act, msa_mask]); # residual.shape = (N_res, N_res, c_z)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': outer_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
    pair_act = tf.keras.layers.Add()([pair_act, residual]); # pair_act.shape = (N_res, N_res, c_z)
  residual = MSARowAttentionWithPairBias(c_m, c_z, num_head = row_num_head)([msa_act, msa_mask, pair_act]); # residual.shape = (N_res, N_res, c_m)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': row_drop_rate})(residual); # residual.shape = (N_res, N_res, c_m)
  msa_act = tf.keras.layers.Add()([msa_act, residual]); # msa_act.shape = (N_res, N_res, c_m)
  if not is_extra_msa:
    residual = MSAColumnAttention(c_m, num_head = column_num_head)([msa_act, msa_mask]); # residual.shape = (N_res, N_res, c_m)
  else:
    residual = MSAColumnGlobalAttention(c_m, num_head = column_num_head)([msa_act, msa_mask]); # residual.shape = (N_res, N_res, c_m)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (tf.shape(x)[0], 1, tf.shape(x)[2])), arguments = {'r': column_drop_rate})(residual); # residual.shape = (N_res, N_res, c_m)
  msa_act = tf.keras.layers.Add()([msa_act, residual]); # msa_act.shape = (N_res, N_res, c_m)
  residual = Transition(c_m, num_intermediate_factor = transition_factor)([msa_act, msa_mask]); # residual.shape = (N_res, N_res, c_m)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': transition_drop_rate})(residual); # residual.shape = (N_res, N_res, c_m)
  msa_act = tf.keras.layers.Add()([msa_act, residual]); # msa_act.shape = (N_res, N_res, c_m)
  if not outer_first:
    residual = OuterProductMean(c_z, c_m, num_outer_channel = outer_num_channel)([msa_act, msa_mask]); # residual.shape = (N_res, N_res, c_z)
    residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': outer_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
    pair_act = tf.keras.layers.Add()([pair_act, residual]); # pair_act.shape = (N_res, N_res, c_z)
  residual = TriangleMultiplication(c_z, intermediate_channel = tri_mult_intermediate_channel, mode = 'outgoing')([pair_act, pair_mask]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': tri_mult_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act = tf.keras.layers.Add()([pair_act, residual]); # pair_act.shape = (N_res, N_res, c_z)
  residual = TriangleMultiplication(c_z, intermediate_channel = tri_mult_intermediate_channel, mode = 'incoming')([pair_act, pair_mask]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': tri_mult_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act = tf.keras.layers.Add()([pair_act, residual]); # pair_act.shape = (N_res, N_res, c_z)
  residual = TriangleAttention(c_z, num_head = num_head)([pair_act, pair_mask]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': tri_attn_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act = tf.keras.layers.Add()([pair_act, residual]); # pair_act.shape = (N_res, N_res, c_z)
  residual = TriangleAttention(c_z, num_head = num_head)([pair_act, pair_mask]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (tf.shape(x)[0], 1, tf.shape(x)[2])), arguments = {'r': tri_attn_drop_rate})(residual); # residual.shape = (N_res, N_res, c_z)
  pair_act = tf.keras.layers.Add()([pair_act, residual]); # pair_act.shape = (N_res, N_res, c_z)
  residual = Transition(c_z, num_intermediate_factor = transition_factor)([pair_act, pair_mask]); # residual.shape = (N_res, N_res, c_z)
  residual = tf.keras.layers.Lambda(lambda x, r: tf.nn.dropout(x, rate = r, noise_shape = (1, tf.shape(x)[1], tf.shape(x)[2])), arguments = {'r': transition_drop_rate})(residual); # residual.shape = (N_Res, N_res, c_z)
  pair_act = tf.keras.layers.Add()([pair_act, residual]); # pair_act.shape = (N_res, N_res, c_z)
  return tf.keras.Model(inputs = inputs, outputs = (msa_act, pair_act));

def make_canonical_transform():
  n_xyz = tf.keras.Input((3,)); # n_xyz.shape = (batch, 3)
  ca_xyz = tf.keras.Input((3,)); # ca_xyz.shape = (batch, 3)
  c_xyz = tf.keras.Input((3,)); # c_xyz.shape = (batch, 3)
  inputs = (n_xyz, ca_xyz, c_xyz);
  n_xyz = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([n_xyz, ca_xyz]); # n_xyz.shape = (batch, 3)
  c_xyz = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([c_xyz, ca_xyz]); # c_xyz.shape = (batch, 3)
  sin_c1 = tf.keras.layers.Lambda(lambda x: -x[:,1] / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1])))(c_xyz); # sin_c1.shape = (batch)
  cos_c1 = tf.keras.layers.Lambda(lambda x: x[:,0] / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1])))(c_xyz); # cos_c1.shape = (batch)
  c1_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([x[1], -x[0], tf.zeros_like(x[0])], axis = -1),
                                                             tf.stack([x[0], x[1], tf.zeros_like(x[0])], axis = -1),
                                                             tf.stack([tf.zeros_like(x[0]), tf.zeros_like(x[0]), tf.ones_like(x[0])], axis = -1)], axis = -2))([sin_c1, cos_c1]); # c1_rot_matrix.shape = (batch, 3, 3)
  sin_c2 = tf.keras.layers.Lambda(lambda x: x[:,2] / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(c_xyz); # sin_c2.shape = (batch)
  cos_c2 = tf.keras.layers.Lambda(lambda x: tf.math.sqrt(tf.math.square(x[:,0]) + tf.math.square(x[:,1])) / tf.math.sqrt(1e-20 + tf.math.square(x[:,0]) + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(c_xyz); # cos_c2.shape = (batch)
  c2_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([x[1], tf.zeros_like(x[0]), x[0]], axis = -1),
                                                             tf.stack([tf.zeros_like(x[0]), tf.ones_like(x[0]), tf.zeros_like(x[0])], axis = -1),
                                                             tf.stack([-x[0], tf.zeros_like(x[0]), x[1]], axis = -1)], axis = -2))([sin_c2, cos_c2]); # c2_rot_matrix.shape = (batch, 3, 3)
  c_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([c2_rot_matrix, c1_rot_matrix]); # c_rot_matrix.shape = (batch, 3, 3)
  n_xyz = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.linalg.matmul(x[0], tf.expand_dims(x[1], axis = -1)), axis = -1))([c_rot_matrix, n_xyz]); # n_xyz.shape = (batch, 3)
  sin_n = tf.keras.layers.Lambda(lambda x: -x[:,2] / tf.math.sqrt(1e-20 + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(n_xyz); # sin_n.shape = (batch)
  cos_n = tf.keras.layers.Lambda(lambda x: x[:,1] / tf.math.sqrt(1e-20 + tf.math.square(x[:,1]) + tf.math.square(x[:,2])))(n_xyz); # cos_n.shape = (batch)
  n_rot_matrix = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([tf.ones_like(x[0]), tf.zeros_like(x[0]), tf.zeros_like(x[0])], axis = -1),
                                                            tf.stack([tf.zeros_like(x[0]), x[1], -x[0]], axis = -1),
                                                            tf.stack([tf.zeros_like(x[0]), x[0], x[1]], axis = -1)], axis = -2))([sin_n, cos_n]); # n_rot_matrix.shape = (batch, 3, 3)
  rot_matrix = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([n_rot_matrix, c_rot_matrix]); # rot_matrix.shape = (batch, 3, 3)
  translation = tf.keras.layers.Lambda(lambda x: -x)(ca_xyz); # translation.shape = (batch, 3)
  return tf.keras.Model(inputs = inputs, outputs = (translation, rot_matrix));

def rot_to_quat(unstack_inputs = False):
  if unstack_inputs:
    rot = tf.keras.Input((3, 3)); # rot.shape = (N_res, 3, 3)
    inputs = (rot,)
    rot = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,2,0)))(rot); # rot.shape = (3, 3, N_res)
  else:
    rot = tf.keras.Input((3, None), batch_size = 3); # rot.shape = (3, 3, N_res)
    inputs = (rot,)
  k = tf.keras.layers.Lambda(lambda x: 1/3 * tf.stack([tf.stack([x[0,0] + x[1,1] + x[2,2], x[2,1] - x[1,2], x[0,2] - x[2,0], x[1,0] - x[0,1]], axis = -1),
                                                       tf.stack([x[2,1] - x[1,2], x[0,0] - x[1,1] - x[2,2], x[0,1] + x[1,0], x[0,2] + x[2,0]], axis = -1),
                                                       tf.stack([x[0,2] - x[2,0], x[0,1] + x[1,0], x[1,1] - x[0,0] - x[2,2], x[1,2] + x[2,1]], axis = -1),
                                                       tf.stack([x[1,0] - x[0,1], x[0,2] + x[2,0], x[1,2] + x[2,1], x[2,2] - x[0,0] - x[1,1]], axis = -1)], axis = -2))(rot); # x.shape = (N_res, 4, 4)
  qs = tf.keras.layers.Lambda(lambda x: tf.linalg.eigh(x)[1])(k); # qs.shape = (N_res, 4, 4)
  # NOTE: return the eigvector of the biggest eigvalue
  qs = tf.keras.layers.Lambda(lambda x: x[...,-1])(qs); # qs.shape = (N_res, 4)
  return tf.keras.Model(inputs = inputs, outputs = qs);

def quat_to_rot():
  normalized_quat = tf.keras.Input((4,)); # normalized_quat.shape = (N_res, 4)
  QUAT_TO_ROT = tf.keras.layers.Lambda(lambda x: tf.stack([tf.stack([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], [[0.,0.,0.],[0.,0.,-2.],[0.,2.,0.]], [[0.,0.,2.],[0.,0.,0.],[-2.,0.,0.]], [[0.,-2.,0.],[2.,0.,0.],[0.,0.,0.]]], axis = 0),
                                                           tf.stack([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], [[0.,2.,0.],[2.,0.,0.],[0.,0.,0.]], [[0.,0.,2.],[0.,0.,0.],[2.,0.,0.]]], axis = 0),
                                                           tf.stack([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]], [[0.,0.,0.],[0.,0.,2.],[0.,2.,0.]]], axis = 0),
                                                           tf.stack([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]]], axis = 0),], axis = 0))(normalized_quat); # QUAT_TO_ROT.shape = (4,4,3,3)
  rot_tensor = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(
                                                  tf.reshape(x[0], (4,4,9)) * 
                                                  tf.reshape(x[1], (tf.shape(x[1])[0], x[1].shape[1], 1, 1)) * # shape = (N_res, 4, 1, 1)
                                                  tf.reshape(x[1], (tf.shape(x[1])[0], 1, x[1].shape[1], 1)),  # shape = (N_res, 1, 4, 1)
                                                  axis = (-3, -2)
                                                ))([QUAT_TO_ROT, normalized_quat]); # rot_tensor.shape = (N_res, 9)
  rot = tf.keras.layers.Reshape((3,3))(rot_tensor); # rot.shape = (N_res, 3, 3)
  return tf.keras.Model(inputs = normalized_quat, outputs = rot);

def invert_point(extra_dims = 0):
  rotation = tf.keras.Input((3, None), batch_size = 3); # rotation.shape = (3, 3, N_res)
  translation = tf.keras.Input((None,), batch_size = 3); # translation.shape = (3, N_res)
  transformed_points = tf.keras.Input((1, None), batch_size = 3); # points.shape = (3, 1, N_res)
  inputs = (rotation, translation, transformed_points);
  for _ in range(extra_dims):
    rotation = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(rotation); # rotation.shape = (3, 3, N_res, 1)
    translation = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(translation); # translation.shape = (3, N_res, 1)
  rot_point = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([transformed_points, translation]); # rot_point.shape = (3, N_res, N_res)
  results = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.linalg.matmul(x[0], x[1], transpose_a = True), axis = -1))([rot_point, translation]); # results.shape = (3, N_res)
  return tf.keras.Model(inputs = inputs, outputs = results);

def SingleTemplateEmbedding(c_z, min_bin = 3.25, max_bin = 50.75, num_bins = 39):
  query_embedding = tf.keras.Input((None, c_z)); # query_embedding.shape = (N_res, N_res, c_z)
  template_aatype = tf.keras.Input(()); # template_aatype.shape = (N_res,)
  template_all_atom_positions = tf.keras.Input((atom_type_num, 3)); # template_all_atom_positions.shape = (N_res, atom_type_num, 3)
  template_all_atom_masks = tf.keras.Input((atom_type_num)); # template_all_atom_masks.shape = (N_res, atom_type_num)
  template_pseudo_beta_mask = tf.keras.Input(()); # template_pseudo_beta_mask.shape = (N_res)
  template_pseudo_beta = tf.keras.Input((None,)); # template_seudo_beta.shape = (N_res, None)
  template_pseudo_beta_mask = tf.keras.Input(()); # template_seudo_beta_mask.shape = (N_res)
  template_mask_2d = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.cast(tf.expand_dims(x[0], axis = 1) * tf.expand_dims(x[0], axis = 0), dtype = x[1].dtype), axis = -1))([template_pseudo_beta_mask, query_embedding]); # template_mask_2d.shape = (N_res, N_res, 1)
  template_dgram = dgram_from_positions(min_bin, max_bin, num_bins)(template_pseudo_beta); # template_dgram.shape = (N_res, N_res, num_bins)
  template_dgram = tf.keras.layers.Lambda(lambda x: tf.cast(x[0], dtype = x[1].dtype))([template_dgram, query_embedding]); # template_dgram.shape = (N_res, N_res, num_bins)
  to_concat = [template_dgram, template_mask_2d];
  aatype = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, 22))(template_aatype); # aatype.shape = (N_res, 22)
  aatype_tile0 = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x[0], axis = 0), (tf.shape(x[1])[0],1,1)))([aatype, template_aatype]); # aatype_tile0.shape = (N_res, N_res, 22)
  aatype_tile1 = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x[0], axis = 1), (1,tf.shape(x[1])[0],1)))([aatype, template_aatype]); # aatype_tile1.shape = (N_res, N_res, 22)
  to_concat.append(aatype_tile0);
  to_concat.append(aatype_tile1);
  n_xyz = tf.keras.layers.Lambda(lambda x, n: x[:,n], arguments = {'n': residue_constants.atom_order['N']})(template_all_atom_positions); # n_xyz.shape = (N_res, 3)
  ca_xyz = tf.keras.layers.Lambda(lambda x, n: x[:,n], arguments = {'n': residue_constants.atom_order['CA']})(template_all_atom_positions); # ca_xyz.shape = (N_res, 3)
  c_xyz = tf.keras.layers.Lambda(lambda x, n: x[:,n], arguments = {'n': residue_constants.atom_order['C']})(template_all_atom_positions); # c_xyz.shape = (N_res, 3)
  translation, rot_matrix = make_canonical_transform()([n_xyz, ca_xyz, c_xyz]); # translation.shape = (N_res, 3) rot_matrix.shape = (N_res, 3, 3)
  # INFO: get inverse transformation (rotation, translation)
  trans = tf.keras.layers.Lambda(lambda x: -x)(translation); # trans.shape = (N_res, 3)
  rot = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1)))(rot_matrix); # rot.shape = (N_res, 3, 3)
  quaternion = rot_to_quat(unstack_inputs = True)(rot); # quaternion.shape = (N_res, 4)
  translation = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(trans); # translation.shape = (3, N_res)
  rotation = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,2,0)))(rot); # rotation.shape = (3,3,N_res)
  points = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -2))(translation); # points.shape = (3, 1, N_res)
  affine_vec = invert_point(extra_dims = 1)([rotation, translation, points]); # affine_vec.shape = (3, N_res)
  inv_distance_scalar = tf.keras.layers.Lambda(lambda x: tf.math.rsqrt(1e-6 + tf.math.reduce_sum(tf.math.square(x), axis = 0)))(affine_vec); # inv_distance_scalar.shape = (N_res)
  template_mask = tf.keras.layers.Lambda(lambda x, n, ca, c: x[..., n] * x[..., ca] * x[..., c], 
                                         arguments = {'n': residue_constants.atom_order['N'], 'ca': residue_constants.atom_order['CA'], 'c': residue_constants.atom_order['C']})(template_all_atom_masks); # template_mask.shape = (N_res)
  template_mask_2d = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 1) * tf.expand_dims(x, axis = 0))(template_mask); # template_mask_2d.shape = (N_res, N_res)
  # TODO

def TemplateEmbedding(c_z):
  query_embedding = tf.keras.Input((None, c_z)); # query_embedding.shape = (N_res, N_res, c_z)
  template_mask = tf.keras.Input(()); # template_mask.shape = (N_template)
  mask_2d = tf.keras.Input((None,)); # mask_2d.shape = (N_res, N_res)
  inputs = (query_embedding, template_mask, mask_2d)
  template_mask = tf.keras.layers.Lambda(lambda x: tf.cast(x[0], dtype = x[1].dtype))([template_mask, query_embedding]); # template_mask.shape = (N_template)
  # TODO

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
  inputs = (target_feat, msa_feat, msa_mask, seq_mask, aatype, prev_pos, prev_msa_first_row, prev_pair, residue_index, extra_msa, extra_msa_mask, extra_has_deletion, extra_deletion_value,);

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
    prev_msa_first_row = tf.keras.layers.LayerNormalization()(prev_msa_first_row); # prev_msa_first_row.shape = (N_res, msa_channel)
    msa_activations = tf.keras.layers.Lambda(lambda x: tf.concat([tf.expand_dims(x[1], axis = 0), tf.zeros((tf.shape(x[0])[0] - 1, tf.shape(x[0])[1], tf.shape(x[0])[2]))], axis = 0) + x[0])([msa_activations, prev_msa_first_row]); # msa_activations.shape = (N_seq, N_res, msa_channel)
    prev_pair = tf.keras.layers.LayerNormalization()(prev_pair); # prev_pair.shape = (N_res, N_res, pair_channel)
    pair_activations = tf.keras.layers.Add()([pair_activations, prev_pair]); # pair_activations.shape = (N_res, N_res, pair_channel)
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
  return tf.keras.Model(inputs = inputs,
                        outputs = (single_activations, pair_activations, msa, single_msa_activations));

def AlphaFoldIteration(num_ensemble, ensemble_representations = False, return_representations = False, c_m = 22, c_z = 25, msa_channel = 256, pair_channel = 128, recycle_pos = True, prev_pos_min_bin = 3.25, prev_pos_max_bin = 20.75, prev_pos_num_bins = 15, recycle_features = True, max_relative_feature = 32, template_enabled = False, extra_msa_channel = 64, extra_msa_stack_num_block = 4, evoformer_num_block = 48, seq_channel = 384,
                       head_masked_msa_output_num = 23,
                       head_distogram_first_break = 2.3125, head_distogram_last_break = 21.6875, head_distogram_num_bins = 64, head_distogram_weight = 0.3):
  target_feat = tf.keras.Input((None, c_m,), batch_siz = num_ensemble); # target_feat.shape = (num_ensemble, N_res, c_m)
  msa_feat = tf.keras.Input((None, None, c_z), batch_siz = num_ensemble); # msa_feat.shape = (num_ensemble, N_seq, N_res, c_z)
  msa_mask = tf.keras.Input((None, None,), batch_siz = num_ensemble); # msa_mask.shape = (num_ensemble, N_seq, N_res)
  seq_mask = tf.keras.Input((None,), batch_siz = num_ensemble); # seq_mask.shape = (num_ensemble, N_res)
  aatype = tf.keras.Input((None,), batch_siz = num_ensemble); # aatype.shape = (num_ensemble, N_res)
  prev_pos = tf.keras.Input((None, atom_type_num, 3), batch_siz = num_ensemble); # prev_pos.shape = (num_ensemble, N_res, atom_type_num, 3)
  prev_msa_first_row = tf.keras.Input((None, msa_channel,), batch_siz = num_ensemble); # prev_msa_first_row.shape = (num_ensemble, N_res, msa_channel)
  prev_pair = tf.keras.Input((None, None, pair_channel), batch_siz = num_ensemble); # prev_pair.shape = (num_ensemble, N_res, N_res, pair_channel)
  residue_index = tf.keras.Input((None,), dtype = tf.int32, batch_siz = num_ensemble); # residue_index.shape = (num_ensemble, N_res)
  extra_msa = tf.keras.Input((None, None,), dtype = tf.int32, batch_siz = num_ensemble); # extra_msa.shape = (num_ensemble, N_seq, N_res)
  extra_msa_mask = tf.keras.Input((None, None,), batch_siz = num_ensemble); # extra_msa_mask.shape = (num_ensemble, N_seq, N_res)
  extra_has_deletion = tf.keras.Input((None, None,), batch_siz = num_ensemble); # extra_has_deletion.shape = (num_ensemble, N_seq, N_res)
  extra_deletion_value = tf.keras.Input((None, None,), batch_siz = num_ensemble); # extra_deletion_value.shape = (num_ensemble, N_seq, N_res)
  seq_length = tf.keras.Input((None,), batch_siz = num_ensemble); # seq_length.shape = (num_ensemble, N_res,)
  if ensemble_representations == False:
    assert num_ensemble == 1;
  inputs = (target_feat, msa_feat, msa_mask, seq_mask, aatype, prev_pos, prev_msa_first_row, prev_pair, residue_index, extra_msa, extra_msa_mask, extra_has_deletion, extra_deletion_value, seq_length);
  embeddings_and_evoformer = EmbeddingsAndEvoformer(c_m, c_z, msa_channel, pair_channel, recycle_pos, prev_pos_min_bin, prev_pos_max_bin, prev_pos_num_bins, recycle_features, max_relative_feature, template_enabled, extra_msa_channel, extra_msa_stack_num_block, evoformer_num_block, seq_channel);
  # iteration 0
  def slice_batch(inputs, n):
    outputs = list();
    for _input in inputs:
      output = tf.keras.layers.Lambda(lambda x, i: x[i], arguments = {'i': n})(_input);
      outputs.append(output);
    return tuple(outputs);
  batch0_inputs = slice_batch([target_feat, msa_feat, msa_mask, seq_mask, aatype, prev_pos, prev_msa_first_row, prev_pair, residue_index, extra_msa, extra_msa_mask, extra_has_deletion, extra_deletion_value], 0);
  representation_update = embeddings_and_evoformer(batch0_inputs);
  if ensemble_representations:
    # NOTE: save msa results
    msa_representation = representation_update[2];
    # iteration 1 to num_ensemble
    for i in range(1, num_ensemble):
      representation_current = representation_update;
      batchi_inputs = slice_batch([target_feat, msa_feat, msa_mask, seq_mask, aatype, prev_pos, prev_msa_first_row, prev_pair, residue_index, extra_msa, extra_msa_mask, extra_has_deletion, extra_deletion_value], i);
      representation = embeddings_and_evoformer(batchi_inputs);
      representation_update = list();
      for current, update in zip(representation_current, representation):
        rep = tf.keras.layers.Add()([current, update]);
        representation_update.append(rep);
    # average on batch
    single_activations, pair_activations, msa, single_msa_activations = representation_update;
    single_activations = tf.keras.layers.Lambda(lambda x, b: x / b, arguments = {'b': num_ensemble})(single_activations);
    pair_activations = tf.keras.layers.Lambda(lambda x, b: x / b, arguments = {'b': num_ensemble})(pair_activations);
    # NOTE: restore msa results
    msa = msa_representation;
    single_msa_activations = tf.keras.layers.Lambda(lambda x, b: x / b, arguments = {'b': num_ensemble})(single_msa_activations);
  else:
    single_activations, pair_activations, msa, single_msa_activations = representation_update;
  batch = batch0_inputs;
  # single_msa_activations.shape = (N_res, msa_channel), pair_activations.shape = (N_seq, N_res, pair_channel), msa.shape = (N_seq, N_res, msa_channel), single_activations.shape = (N_res, seq_channel)
  # connect to heads
  masked_msa = MaskedMsaHead(msa_channel, head_masked_msa_output_num)(msa); # masked_msa.shape = (N_seq, N_seq, head_masked_msa_output_num);
  distogram_logits, distogram_breaks = DistogramHead(pair_channel, head_distogram_num_bins, head_distogram_first_break, head_distogram_last_break)(pair_activations); # distogram_logits.shape = (N_res, N_res, head_distogram_num_bins)
  

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
  rot = np.random.normal(size = (4, 3, 3));
  quat = rot_to_quat(True)(rot);
  print(quat.shape);
  rot = np.random.normal(size = (3, 3, 4));
  quat = rot_to_quat(False)(rot);
  print(quat.shape);
  quat = np.random.normal(size = (10, 4)); #N_template, atom_type_num, 4
  rot = quat_to_rot()(quat);
  print(rot.shape);
  points = np.random.normal(size = (3,1,4));
  rotation = np.random.normal(size = (3, 3, 4));
  translation = np.random.normal(size = (3, 4));
  results = invert_point(extra_dims = 1)([rotation, translation, points]);
  print(results.shape);
