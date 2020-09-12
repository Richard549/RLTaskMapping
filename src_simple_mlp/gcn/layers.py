from .inits import *
import tensorflow as tf
import logging

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
		"""Helper function, assigns unique layer IDs."""
		if layer_name not in _LAYER_UIDS:
				_LAYER_UIDS[layer_name] = 1
				return 1
		else:
				_LAYER_UIDS[layer_name] += 1
				return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
		"""Dropout for sparse tensors."""
		random_tensor = keep_prob
		random_tensor += tf.random_uniform(noise_shape)
		dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
		pre_out = tf.sparse_retain(x, dropout_mask)
		return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
		"""Wrapper for tf.matmul (sparse vs dense)."""
		if sparse:
				res = tf.sparse_tensor_dense_matmul(x, y)
		else:
				res = tf.matmul(x, y)
		return res


class Layer(object):
		"""Base layer class. Defines basic API for all layer objects.
		Implementation inspired by keras (http://keras.io).

		# Properties
				name: String, defines the variable scope of the layer.
				logging: Boolean, switches Tensorflow histogram logging on/off

		# Methods
				_call(inputs): Defines computation graph of layer
						(i.e. takes input, returns output)
				__call__(inputs): Wrapper for _call()
				_log_vars(): Log all variables
		"""

		def __init__(self, **kwargs):
				allowed_kwargs = {'name', 'logging'}
				for kwarg in kwargs.keys():
						assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
				name = kwargs.get('name')
				if not name:
						layer = self.__class__.__name__.lower()
						name = layer + '_' + str(get_layer_uid(layer))
				self.name = name
				self.vars = {}
				logging = kwargs.get('logging', False)
				self.logging = logging
				self.sparse_inputs = False

		def _call(self, inputs):
				return inputs

		def __call__(self, inputs):
				with tf.name_scope(self.name):
						if self.logging and not self.sparse_inputs:
								tf.summary.histogram(self.name + '/inputs', inputs)
						outputs = self._call(inputs)
						if self.logging:
								tf.summary.histogram(self.name + '/outputs', outputs)
						return outputs

		def _log_vars(self):
				for var in self.vars:
						tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
		"""Graph convolution layer."""
		def __init__(self, input_dim, output_dim, placeholders, depth, model, variable_support, dropout=0.,
								 sparse_inputs=False, act=tf.nn.relu, bias=False,
								 featureless=False, **kwargs):
				super(GraphConvolution, self).__init__(**kwargs)

				if dropout:
						self.dropout = placeholders['dropout']
				else:
						self.dropout = 0.

				self.model = model
				self.depth = depth
				self.act = act
				self.supports = placeholders['supports']
				self.placeholders = placeholders
				self.sparse_inputs = sparse_inputs
				self.featureless = featureless
				self.bias = bias
				self.variable_support = variable_support

				with tf.variable_scope(self.name + '_vars'):
						self.vars['weights_0'] = glorot([input_dim, output_dim], name='weights_0')
						#self.vars['weights_0'] = ones([input_dim, output_dim], name='weights_0')
						if self.bias:
								self.vars['bias'] = zeros([output_dim], name='bias')

				if self.logging:
						self._log_vars()

		def _call(self, inputs):

			#self.model.layer_print_ops.append(tf.print("GCN layer at depth " + str(self.depth) + " inputs:\n", inputs, summarize=-1))

			logging.info("")
			logging.info("GCN inputs are:" + str(inputs))
			if self.variable_support:
			
				sup = self.supports

				#pre_sup = dot(inputs, self.vars['weights_0'], sparse=False)
				pre_sup = tf.matmul(inputs, self.vars['weights_0'])

				#self.model.layer_print_ops.append(tf.print("GCN layer (variable_support) at depth " + str(self.depth) + " matmul(x, weights):\n", pre_sup, summarize=-1))

				#output = dot(sup, pre_sup, sparse=True)
				output = tf.sparse_tensor_dense_matmul(sup, pre_sup)
				
				#self.model.layer_print_ops.append(tf.print("GCN layer (variable_support) at depth " + str(self.depth) + " matmul(support, matmul(x, weights)):\n", output, summarize=-1))

				return output

			else:

				overall_output = []

				xs = tf.split(inputs, FLAGS.num_simultaneous_graphs, axis=0)
				#supports = tf.split(self.supports, FLAGS.num_simultaneous_graphs, axis=0) # TODO this is what it was before sparseness
				#supports = tf.sparse_split(sp_input=self.supports, num_split=FLAGS.num_simultaneous_graphs, axis=0)

				overall_output = []
				
				for x in xs:
					
					pre_sup = tf.matmul(x, self.vars['weights_0'])
				
					#self.model.layer_print_ops.append(tf.print("GCN layer at depth " + str(self.depth) + " matmul(x, weights):\n", pre_sup, summarize=-1))

					output = tf.sparse_tensor_dense_matmul(self.supports, pre_sup)
					
					#self.model.layer_print_ops.append(tf.print("GCN layer at depth " + str(self.depth) + " matmul(support, matmul(x, weights)):\n", output, summarize=-1))
				
					# bias
					if self.bias:
							output += self.vars['bias']
					
					activated_output = self.act(output)

					overall_output.append(activated_output)

				overall_output = tf.concat(overall_output, axis=0)
				logging.info("GCN output:" + str(overall_output))
				return overall_output

class Dense(Layer):
		"""Dense layer."""
		def __init__(self, input_dim, output_dim, placeholders, depth, model, first_pooling=False, final_policy_layer=False, dropout=0., sparse_inputs=False,
								 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
				super(Dense, self).__init__(**kwargs)

				if dropout:
						self.dropout = placeholders['dropout']
				else:
						self.dropout = 0.

				self.model = model
				self.first_pooling = first_pooling
				self.act = act
				self.sparse_inputs = sparse_inputs
				self.featureless = featureless
				self.bias = bias
				self.placeholders = placeholders
				self.input_dim = input_dim
				self.output_dim = output_dim
				self.final_policy_layer = final_policy_layer

				if first_pooling:
					# and self.model.method_type == "batched_statevalue":
					input_dim *= self.model.num_nodes_per_graph

				logging.info("Layer " + str(depth) + " input dim is " + str(input_dim))
				logging.info("Layer " + str(depth) + " output dim is " + str(output_dim))

				with tf.variable_scope(self.name + '_vars'):
						self.vars['weights_0'] = glorot([input_dim, output_dim],
																					name='weights_0')
						if self.bias:
								self.vars['bias_0'] = zeros([output_dim], name='bias')

				if self.logging:
						self._log_vars()

		def _call(self, inputs):
				logging.info("")
				logging.info("Dense inputs are:" + str(inputs))
				logging.info("Layer weights are:" + str(self.vars['weights_0']))

				logging.info("Num graphs = " + str(FLAGS.num_simultaneous_graphs))
				logging.info("Input dim = " + str(self.input_dim))
				logging.info("Num nodes per graph = " + str(self.model.num_nodes_per_graph))

				if self.first_pooling:

					# input shape is (num_graphs * num_nodes_per_graph, hidden GCN neurons)
					# i.e. each node produces a vector of length (hidden GCN)
					# for each graph, I need this to be concatenated as a vector of length (GCN_hidden neurons * num_nodes_per_graph)
					# then multiplied by the weights with output dim = dense_hidden neurons
					# then concatenated to give output of shape (num_graphs, dense_hidden neurons)

					inputs_per_graph = tf.split(inputs, FLAGS.num_simultaneous_graphs, axis=0)
					
					overall_output = []
					for inputs in inputs_per_graph:

						# now reshape this to a vector
						x	= tf.reshape(inputs, [1,-1])

						weights = self.vars['weights_0']
						output = tf.matmul(x, weights)
						overall_output.append(output)

					output = tf.concat(overall_output, axis=0)

				else:
					if self.model.method_type == "ppo_policy" or self.model.method_type == "reinforce_policy":

						xs = tf.split(inputs, FLAGS.num_simultaneous_graphs)
		
						overall_output = []

						for x in xs:

							logging.info("x is " + str(x))

							output = tf.matmul(x, self.vars['weights_0'])
							#if self.final_policy_layer:
							#	output = tf.reshape(output, [self.model.num_nodes_per_graph, self.model.num_actions])
							overall_output.append(output)
						
						output = tf.concat(overall_output,axis=0)
						logging.info("output from " + str(self.model.method_type) + " layer: " + str(output))
						logging.info("")

					elif self.model.method_type == "batched_statevalue" or self.model.method_type == "rnd_predictor" or self.model.method_type == "rnd_target":
						xs = tf.split(inputs, FLAGS.num_simultaneous_graphs, axis=0)

						overall_output = []
						for x in xs:
							logging.info("x is " + str(x))
							logging.info("weights vector is " + str(self.vars['weights_0']))
							output = tf.squeeze(tf.matmul(x, self.vars['weights_0']))
							overall_output.append(output)

						output = tf.stack(overall_output)

						logging.info("output from SV Dense layer:" + str(output))

				return self.act(output)

'''
if self.first_pooling:
	#and self.model.method_type == "batched_statevalue": # then we are taking a set of values from each node, and operating on that vector
	logging.info("This is the first " + str(self.model.method_type) + " pooling layer.")
	
	xs = tf.dynamic_partition(inputs, self.placeholders['nodes_mask'], FLAGS.num_simultaneous_graphs+1)[1:]

	#xs = tf.split(inputs, FLAGS.num_simultaneous_graphs, axis=0)
	#actioned_mask = tf.dynamic_partition(self.placeholders['nodes_mask'], self.placeholders['nodes_mask'], FLAGS.num_simultaneous_graphs+1)[1:] # this does NOT work! because it only returns the 1s from each, not the actual mask of 0s and 1s for each
	actioned_mask = tf.split(self.placeholders['nodes_mask'], FLAGS.num_simultaneous_graphs)

	overall_output = []
	for x, mask in zip(xs, actioned_mask):
	
		self.model.layer_print_ops.append(tf.print("mask is:",tf.shape(mask), ":", mask))
		logging.info("x: " + str(x)) # this should be (NUM_POST_FILTER NODES_FOR_GRAPH, INPUT_DIM)

		x = tf.reshape(x, [1,-1]) # flattened to a row-vector (1, NUM_POST_FILTER_NODES_FOR_GRAPH * INPUT_DIM) that represents the particular graph

		logging.info("weights:" + str(self.vars['weights_0']))

		mask = tf.reshape(mask, [-1, 1])
		mask = tf.tile(mask, [1,self.input_dim])
		mask = tf.reshape(mask, [-1])

		self.model.layer_print_ops.append(tf.print("new mask is:",tf.shape(mask), ":", mask, summarize=-1))

		#weights = tf.dynamic_partition(self.vars['weights_0'], mask, 2)[1] # need to change all non-zeros to 1 if we use this
		weights = tf.boolean_mask(self.vars['weights_0'], mask)
		logging.info("weights:" + str(weights))
		
		# weights is a 2 dimensional (NUM_POST_FILTER_NODES_FOR_GRAPH, OUTPUT_DIM) matrix

		#output = tf.matmul(tf.transpose(x), weights)
		#self.model.layer_print_ops.append(tf.print("x is:",x))
		#self.model.layer_print_ops.append(tf.print("weights is:",weights))

		self.model.layer_print_ops.append(tf.print("x is:",tf.shape(x), ":", x))
		self.model.layer_print_ops.append(tf.print("weights is:",tf.shape(weights), ":", weights)) # this is 16, 32 somehow? 
		#output = tf.matmul(tf.transpose(x), weights)
		output = tf.matmul(x, weights)
		overall_output.append(output)

	output = tf.squeeze(tf.stack(overall_output))

	logging.info("output from first " + str(self.model.method_type) + " Dense pooling layer: " + str(output))
	logging.info("")
	
	self.model.layer_print_ops.append(tf.print("output from " + str(self.model.method_type) + " Dense pooling layer is:",tf.shape(output), ":", output))
'''
