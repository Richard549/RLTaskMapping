from .layers import *
from .metrics import *
import logging

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
	def __init__(self, **kwargs):
		allowed_kwargs = {'name', 'logging'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
		name = kwargs.get('name')
		if not name:
			name = self.__class__.__name__.lower()
		self.name = name

		logging = kwargs.get('logging', False)
		self.logging = logging
		
		self.vars = {}
		self.placeholders = {}

		self.layers = []
		self.activations = []

		self.inputs = None
		self.outputs = None

		self.method_type = FLAGS.method_type

		self.loss = 0
		self.optimizer = None
		self.opt_op = None
		self.masked_prediction_op = None
		self.prediction_op = None

	def _build(self):
		raise NotImplementedError

	def build(self):
		""" Wrapper for _build() """
		with tf.variable_scope(self.name):
			self._build()

		# Build sequential layer model
		self.activations.append(self.inputs)
		logging.info("inputs:" + str(self.inputs))
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			logging.info("hidden:" + str(hidden))
			self.activations.append(hidden)
		self.outputs = self.activations[-1]
		logging.info("outputs:" + str(self.outputs))

		# Store model variables for easy access
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

		self._build_opt()
		
		#self._build_masked_prediction_op()
		self._build_prediction_op()

	def predict(self):
		pass

	def _loss(self):
		raise NotImplementedError

	def save(self, sess=None):
		if not sess:
			raise AttributeError("TensorFlow session not provided.")
		saver = tf.train.Saver(self.vars)
		save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
		print("Model saved in file: %s" % save_path)

	def load(self, sess=None):
		if not sess:
			raise AttributeError("TensorFlow session not provided.")
		saver = tf.train.Saver(self.vars)
		save_path = "tmp/%s.ckpt" % self.name
		saver.restore(sess, save_path)
		print("Model restored from file: %s" % save_path)

class GCN(Model):
	def __init__(self, placeholders, input_dim, num_graphs, num_nodes_per_graph, variable_support, **kwargs):
		super(GCN, self).__init__(**kwargs)

		logging.info("Model name is:" + str(self.name))

		self.inputs = placeholders["features"]

		self.layer_print_ops = []
		self.num_graphs = num_graphs
		self.num_actions = 2
		self.num_nodes_per_graph = num_nodes_per_graph
		self.input_dim = input_dim
		self.variable_support = variable_support

		'''
		if FLAGS.method_type == "batched_statevalue":
			self.output_dim = FLAGS. # a single node+its neighbourhood is represented by a convolved vector of this length
		elif FLAGS.method_type == "rnd_target" or FLAGS.method_type == "rnd_predictor":
			self.output_dim = 8 # the same as batched_statevalue
		elif FLAGS.method_type == "ppo_policy":
			self.output_dim = 8 # a single node+its neighbourhood is represented by a convolved vector of this length
		else:
			raise NotImplementedError()
		'''

		self.placeholders = placeholders

		self.build()

	def _build_opt(self):
		
		self.fit_print_ops = []

		if self.method_type == "reinforce_policy":
			
			network_outputs_per_graph = tf.split(self.predict(), FLAGS.num_simultaneous_graphs)

			action_mask_per_graph = tf.split(tf.one_hot(self.placeholders['actions'], self.num_actions), FLAGS.num_simultaneous_graphs)

			neg_log_probs_per_graph = []

			for network_outputs, action_mask in zip(network_outputs_per_graph, action_mask_per_graph):

				network_probabilities = tf.nn.softmax(network_outputs)
				neg_log_prob = tf.reduce_sum(-tf.log(network_probabilities) * action_mask, axis=1)
				neg_log_probs_per_graph.append(neg_log_prob)

			rewards = self.placeholders['rewards']

			self.loss = tf.reduce_mean(neg_log_probs_per_graph * rewards)

			self.fit_print_ops.append(tf.print("loss:",self.loss))

			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.loss_grads = self.optimizer.compute_gradients(self.loss)
			self.opt_op = self.optimizer.minimize(self.loss)
			
		elif self.method_type == "ppo_policy":
			
			network_outputs = self.predict()

			old_probabilities = self.placeholders['old_probabilities'] # one per graph
			
			self.fit_print_ops.append(tf.print("network_outputs before:",tf.shape(network_outputs),":",network_outputs))

			#network_outputs = tf.dynamic_partition(network_outputs, self.placeholders['nodes_mask'], FLAGS.num_simultaneous_graphs+1)[1:] # this doesn't work like I thought it did!!
			network_outputs = tf.split(network_outputs, FLAGS.num_simultaneous_graphs)

			self.fit_print_ops.append(tf.print("network_outputs after:",tf.shape(network_outputs),":",network_outputs))

			action_masks = tf.split(self.placeholders['actioned_labels_mask'], FLAGS.num_simultaneous_graphs)

			new_probabilities = []

			for outputs, action_mask in zip(network_outputs, action_masks):

				self.fit_print_ops.append(tf.print("ACTIONS MASK:",tf.shape(action_mask),":",action_mask))

				masked_logits = tf.reshape(outputs, [-1])
				probability_distribution = tf.nn.softmax(masked_logits)
				
				self.fit_print_ops.append(tf.print("probability_distribution:",tf.shape(probability_distribution),":",probability_distribution,summarize=-1))
				flat_action_mask = tf.reshape(action_mask, [-1])
				self.fit_print_ops.append(tf.print("flat action mask:",tf.shape(flat_action_mask),":",flat_action_mask,summarize=-1))
			
				logging.info("probability_distribution = " + str(probability_distribution))
				logging.info("action_mask = " + str(action_mask))
				#probability = tf.dynamic_partition(probability_distribution, action_mask, 2)[1]
				probability = tf.dynamic_partition(probability_distribution, flat_action_mask, 2)[1]

				new_probabilities.append(probability)

			#vanilla_loss = -tf.reduce_sum(self.placeholders['rewards'] * tf.log(new_probabilities))

			ratios = tf.exp(tf.log(tf.add(new_probabilities,1e-9)) - tf.log(old_probabilities+1e-9)) # this is more numerically stable, I think?
			rewards = self.placeholders['rewards'] # one per transition
			
			self.fit_print_ops.append(tf.print("Ratios:",ratios))
			self.fit_print_ops.append(tf.print("Rewards:",rewards))

			policy_network_unclipped_loss = ratios * rewards
			policy_network_clipped_loss = tf.clip_by_value(ratios, 1.0-FLAGS.e, 1.0+FLAGS.e) * rewards
			
			minimums = tf.minimum(policy_network_unclipped_loss, policy_network_clipped_loss) # minimize the loss so we are conservative with our updates
			
			self.fit_print_ops.append(tf.print("minimums:",minimums))

			#self.print_op1 = tf.print("old probabilities:", old_probabilities, ". new_probabilities:", new_probabilities, ". Ratios:", ratios, ". Ratios2:", ratios_2, ". Minimums:", minimums, summarize=-1)

			self.loss = - tf.reduce_mean(minimums) # minimizing -loss = maximizing the loss
			
			self.fit_print_ops.append(tf.print("loss:",self.loss))
		
			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

			self.loss_grads = self.optimizer.compute_gradients(self.loss)
			self.opt_op = self.optimizer.minimize(self.loss)
		
		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_predictor":
			
			# I don't need this because we want to calculate the gradients for all parameters (model weights), it just so happens we only calculate loss on a particular set of examples
			# stop gradients for all but the actioned nodes given by nodes_mask
			#mask_h = tf.logical_not(tf.cast(self.placeholders['nodes_mask'], dtype=tf.bool))
			#mask = tf.cast(self.placeholders['nodes_mask'], dtype=self.predict().dtype)
			#mask_h = tf.cast(mask_h, dtype=self.predict().dtype)
			#stopped_gradient_outputs = tf.stop_gradient(mask_h * self.predict()) + mask * self.predict()
			network_outputs = self.predict()

			#outputs = tf.dynamic_partition(network_outputs, self.placeholders['nodes_mask'], FLAGS.num_simultaneous_graphs+1)[1:] # ignore all zeros
			#outputs = tf.dynamic_partition(network_outputs, np.arange(0,FLAGS.num_simultaneous_graphs), FLAGS.num_simultaneous_graphs+1) # ignore all zeros
			
			self.fit_print_ops.append(tf.print("SV outputs:", network_outputs))
			
			logging.info("Network outputs:" + str(network_outputs))
			logging.info("Values:" + str(self.placeholders['values']))

			self.loss = tf.losses.mean_squared_error(network_outputs, self.placeholders['values'])
			self.fit_print_ops.append(tf.print("SV MSE ", self.loss, summarize=-1))

			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.loss_grads = self.optimizer.compute_gradients(self.loss, [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='statevalue_network') if v.name.startswith("statevalue_network/gcn")]
)
			self.opt_op = self.optimizer.minimize(self.loss)

		elif self.method_type == "rnd_target":
			pass # don't construct the optimizer for the target
		else:
			raise RuntimeError("Method type not recognised. Exiting.")

	def _build_prediction_op(self):

		self.pred_print_ops = []

		if self.method_type == "ppo_policy" or self.method_type == "reinforce_policy":
			# assuming only one graph has active nodes_mask corresponding to the actionable nodes
			masked_logits = tf.dynamic_partition(self.predict(), self.placeholders['graphs_mask'], 2)[1]
			
			#self.pred_print_ops.append(tf.print("Policy masked logits has shape ", tf.shape(masked_logits), ":", masked_logits))
			
			masked_shape = tf.shape(masked_logits)
			
			masked_logits = tf.reshape(masked_logits, [-1])
			probability_distribution = tf.nn.softmax(masked_logits)
			#self.pred_print_ops.append(tf.print("Policy direct softmax has shape ", tf.shape(probability_distribution), ":", probability_distribution))

			self.prediction_op = tf.reshape(probability_distribution, masked_shape)
			#self.masked_prediction_op = probability_distribution

		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_target" or self.method_type == "rnd_predictor":

			#masked_logits = tf.boolean_mask(self.predict(), self.placeholders['nodes_mask'])
			#masked_logits = tf.dynamic_partition(self.predict(), tf.layers.flatten(tf.sparse_to_dense(sparse_nodes_mask.indices, sparse_nodes_mask.dense_shape, sparse_nodes_mask.values)), 2)
			
			# assuming only one graph has active nodes_mask
			#masked_logits = tf.dynamic_partition(self.predict(), self.placeholders['nodes_mask'], 2)[1] # corresponding to the ones

			#self.pred_print_ops.append(tf.print("SV masked logits has shape ", tf.shape(masked_logits), ":", masked_logits))

			self.prediction_op = self.predict()

		else:
			raise RuntimeError("Method type not recognised. Exiting.")

	def _build_masked_prediction_op(self):

		self.pred_print_ops = []

		if self.method_type == "ppo_policy" or self.method_type == "reinforce_policy":
			# assuming only one graph has active nodes_mask corresponding to the actionable nodes
			masked_logits = tf.dynamic_partition(self.predict(), self.placeholders['graphs_mask'], 2)[1]
			
			#self.pred_print_ops.append(tf.print("Policy masked logits has shape ", tf.shape(masked_logits), ":", masked_logits))
			
			masked_shape = tf.shape(masked_logits)
			
			masked_logits = tf.reshape(masked_logits, [-1])
			probability_distribution = tf.nn.softmax(masked_logits)
			#self.pred_print_ops.append(tf.print("Policy direct softmax has shape ", tf.shape(probability_distribution), ":", probability_distribution))

			self.masked_prediction_op = tf.reshape(probability_distribution, masked_shape)
			#self.masked_prediction_op = probability_distribution

		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_target" or self.method_type == "rnd_predictor":

			#masked_logits = tf.boolean_mask(self.predict(), self.placeholders['nodes_mask'])
			#masked_logits = tf.dynamic_partition(self.predict(), tf.layers.flatten(tf.sparse_to_dense(sparse_nodes_mask.indices, sparse_nodes_mask.dense_shape, sparse_nodes_mask.values)), 2)
			
			# assuming only one graph has active nodes_mask
			#masked_logits = tf.dynamic_partition(self.predict(), self.placeholders['nodes_mask'], 2)[1] # corresponding to the ones

			#self.pred_print_ops.append(tf.print("SV masked logits has shape ", tf.shape(masked_logits), ":", masked_logits))

			self.masked_prediction_op = self.predict()

		else:
			raise RuntimeError("Method type not recognised. Exiting.")

	def _build(self):

		if self.method_type == "ppo_policy" or self.method_type == "reinforce_policy":

			# GCN 
			self.layers.append(GraphConvolution(input_dim=self.input_dim,
												output_dim=FLAGS.gcn_hidden,
												placeholders=self.placeholders,
												depth=0,
												model=self,
												act=tf.nn.leaky_relu,
												dropout=False,
												variable_support=self.variable_support,
												sparse_inputs=False,
												logging=self.logging
												))

			for hidden_idx in range(FLAGS.num_gcn_layers-2):
				self.layers.append(GraphConvolution(input_dim=FLAGS.gcn_hidden,
													output_dim=FLAGS.gcn_hidden,
													placeholders=self.placeholders,
													depth=hidden_idx+1,
													model=self,
													variable_support=self.variable_support,
													act=tf.nn.leaky_relu,
													dropout=False,
													logging=self.logging
													))

			# GCN
			self.layers.append(GraphConvolution(input_dim=FLAGS.gcn_hidden,
												output_dim=FLAGS.gcn_output, # output_dim neurons per node
												placeholders=self.placeholders,
												depth=FLAGS.num_gcn_layers,
												model=self,
												variable_support=self.variable_support,
												act=tf.nn.leaky_relu,
												dropout=False,
												logging=self.logging
												))

			# flatten and FC
			self.layers.append(Dense(input_dim=FLAGS.gcn_output,
										output_dim=FLAGS.pooling_hidden,
										placeholders=self.placeholders,
										depth=FLAGS.num_gcn_layers+1,
										model=self,
										first_pooling=True,
										act=tf.nn.leaky_relu,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))
			
			for layer_idx in range(FLAGS.num_dense_layers-2):
				self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
											output_dim=FLAGS.pooling_hidden,
											placeholders=self.placeholders,
											depth=FLAGS.num_gcn_layers+layer_idx+2,
											model=self,
											act=tf.nn.leaky_relu,
											dropout=False,
											sparse_inputs=False,
											logging=self.logging
											))

			# FC
			self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
										output_dim=self.num_actions, # simply whether to put the 'next' node on CPU or GPU as a probability map over two neurons
										placeholders=self.placeholders,
										depth=FLAGS.num_dense_layers+1,
										final_policy_layer=True,
										model=self,
										act=lambda x: x,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))

		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_predictor" or self.method_type == "rnd_target":
			
			# TODO the depths are all messed up, sort this out so it is more general
			# GCN 
			self.layers.append(GraphConvolution(input_dim=self.input_dim,
												output_dim=FLAGS.gcn_hidden,
												placeholders=self.placeholders,
												depth=0,
												model=self,
												act=tf.nn.leaky_relu,
												dropout=False,
												variable_support=self.variable_support,
												sparse_inputs=False,
												logging=self.logging
												))

			for layer_idx in range(FLAGS.num_gcn_layers-2):
				self.layers.append(GraphConvolution(input_dim=FLAGS.gcn_hidden,
													output_dim=FLAGS.gcn_hidden,
													placeholders=self.placeholders,
													depth=layer_idx+1,
													model=self,
													variable_support=self.variable_support,
													act=tf.nn.leaky_relu,
													dropout=False,
													logging=self.logging
													))

			# GCN
			self.layers.append(GraphConvolution(input_dim=FLAGS.gcn_hidden,
												output_dim=FLAGS.gcn_output, # output_dim neurons per node
												placeholders=self.placeholders,
												depth=FLAGS.num_gcn_layers,
												model=self,
												variable_support=self.variable_support,
												act=tf.nn.leaky_relu,
												dropout=False,
												logging=self.logging
												))

			# flatten and FC
			self.layers.append(Dense(input_dim=FLAGS.gcn_output,
										output_dim=FLAGS.pooling_hidden,
										placeholders=self.placeholders,
										depth=FLAGS.num_gcn_layers+1,
										model=self,
										first_pooling=True,
										act=tf.nn.leaky_relu,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))
			
			for layer_idx in range(FLAGS.num_dense_layers-2):
				self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
											output_dim=FLAGS.pooling_hidden,
											placeholders=self.placeholders,
											depth=FLAGS.num_gcn_layers+layer_idx+2,
											model=self,
											act=tf.nn.leaky_relu,
											dropout=False,
											sparse_inputs=False,
											logging=self.logging
											))

			# FC
			self.layers.append(Dense(input_dim=FLAGS.pooling_hidden, # take all the node values and reduce them to one via a dense layer
										output_dim=1,
										placeholders=self.placeholders,
										depth=FLAGS.num_gcn_layers+FLAGS.num_dense_layers+1,
										model=self,
										act=lambda x: x,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))


	def predict(self):
		return self.outputs

class MLP(Model):
	def __init__(self, placeholders, input_dim, num_graphs, num_nodes_per_graph, variable_support, **kwargs):
		super(MLP, self).__init__(**kwargs)

		logging.info("Model name is:" + str(self.name))

		self.inputs = placeholders["features"]

		self.layer_print_ops = []
		self.num_graphs = num_graphs
		self.num_actions = 2
		self.num_nodes_per_graph = num_nodes_per_graph
		self.input_dim = input_dim
		self.variable_support = variable_support

		self.placeholders = placeholders

		self.build()

	def _build_opt(self):
		
		self.fit_print_ops = []
			
		if self.method_type == "ppo_policy":
			
			network_outputs = self.predict()

			# get old probability
			old_probabilities = self.placeholders['old_probabilities'] # one per graph
			
			network_outputs = tf.split(network_outputs, FLAGS.num_simultaneous_graphs) # ignore all zeros

			action_masks = tf.split(self.placeholders['actioned_labels_mask'], FLAGS.num_simultaneous_graphs)

			new_probabilities = []

			for outputs, action_mask in zip(network_outputs, action_masks):

				masked_logits = tf.reshape(outputs, [-1])
				probability_distribution = tf.nn.softmax(masked_logits)
				
				flat_action_mask = tf.reshape(action_mask, [-1])
			
				probability = tf.dynamic_partition(probability_distribution, flat_action_mask, 2)[1]

				new_probabilities.append(probability)

			ratios = tf.exp(tf.log(tf.add(new_probabilities,1e-9)) - tf.log(old_probabilities+1e-9)) # this is more numerically stable, I think?
			rewards = self.placeholders['rewards'] # one per transition
			
			policy_network_unclipped_loss = ratios * rewards
			policy_network_clipped_loss = tf.clip_by_value(ratios, 1.0-FLAGS.e, 1.0+FLAGS.e) * rewards
			
			minimums = tf.minimum(policy_network_unclipped_loss, policy_network_clipped_loss) # minimize the loss so we are conservative with our updates
			
			self.loss = - tf.reduce_mean(minimums) # minimizing -loss = maximizing the loss
			
			self.fit_print_ops.append(tf.print("loss:",self.loss))
		
			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

			self.loss_grads = self.optimizer.compute_gradients(self.loss)
			self.opt_op = self.optimizer.minimize(self.loss)
		
		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_predictor":
			
			network_outputs = self.predict()

			self.loss = tf.losses.mean_squared_error(network_outputs, self.placeholders['values'])
			self.fit_print_ops.append(tf.print("SV MSE ", self.loss, summarize=-1))

			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.loss_grads = self.optimizer.compute_gradients(self.loss, [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='statevalue_network') if v.name.startswith("statevalue_network/" + str(self.name))]
)
			self.opt_op = self.optimizer.minimize(self.loss)

		elif self.method_type == "rnd_target":
			pass # don't construct the optimizer for the target
		else:
			raise RuntimeError("Method type not recognised. Exiting.")

	def _build_masked_prediction_op(self):

		self.pred_print_ops = []
		
		if self.method_type == "ppo_policy" or self.method_type == "reinforce_policy":
			# assuming only one graph has active nodes_mask corresponding to the actionable nodes
			masked_logits = tf.dynamic_partition(self.predict(), self.placeholders['graphs_mask'], 2)[1]
			
			#self.pred_print_ops.append(tf.print("Policy masked logits has shape ", tf.shape(masked_logits), ":", masked_logits))
			
			masked_shape = tf.shape(masked_logits)
			
			masked_logits = tf.reshape(masked_logits, [-1])
			probability_distribution = tf.nn.softmax(masked_logits)
			#self.pred_print_ops.append(tf.print("Policy direct softmax has shape ", tf.shape(probability_distribution), ":", probability_distribution))

			self.masked_prediction_op = tf.reshape(probability_distribution, masked_shape)
			#self.masked_prediction_op = probability_distribution

		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_target" or self.method_type == "rnd_predictor":

			self.masked_prediction_op = self.predict()

		else:
			raise RuntimeError("Method type not recognised. Exiting.")

	def _build(self):

		if self.method_type == "ppo_policy":

			# flatten and FC
			self.layers.append(Dense(input_dim=self.input_dim,
										output_dim=FLAGS.pooling_hidden,
										placeholders=self.placeholders,
										depth=0,
										model=self,
										first_pooling=True,
										act=tf.nn.leaky_relu,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))
			
			for layer_idx in range(FLAGS.num_dense_layers-2):
				self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
											output_dim=FLAGS.pooling_hidden,
											placeholders=self.placeholders,
											depth=layer_idx+1,
											model=self,
											act=tf.nn.leaky_relu,
											dropout=False,
											sparse_inputs=False,
											logging=self.logging
											))

			# FC
			self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
										output_dim=self.num_nodes_per_graph * self.num_actions , # flattened vector of logits corresponding to (num_nodes, num_actions), and there will be num_graphs of these vectors
										placeholders=self.placeholders,
										depth=FLAGS.num_dense_layers,
										final_policy_layer=True,
										model=self,
										act=lambda x: x,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))

		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_predictor" or self.method_type == "rnd_target":
			
			# flatten and FC
			self.layers.append(Dense(input_dim=self.input_dim,
										output_dim=FLAGS.pooling_hidden,
										placeholders=self.placeholders,
										depth=0,
										model=self,
										first_pooling=True,
										act=tf.nn.leaky_relu,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))
			
			for layer_idx in range(FLAGS.num_dense_layers-2):
				self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
											output_dim=FLAGS.pooling_hidden,
											placeholders=self.placeholders,
											depth=layer_idx+1,
											model=self,
											act=tf.nn.leaky_relu,
											dropout=False,
											sparse_inputs=False,
											logging=self.logging
											))

			# FC
			self.layers.append(Dense(input_dim=FLAGS.pooling_hidden, # take all the node values and reduce them to one via a dense layer
										output_dim=1,
										placeholders=self.placeholders,
										depth=FLAGS.num_dense_layers,
										model=self,
										act=lambda x: x,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))

	def predict(self):
		return self.outputs

# i.e. no convolution!
class SIMPLE_MLP(Model):
	def __init__(self, placeholders, input_dim, num_graphs, num_nodes_per_graph, variable_support, **kwargs):
		super(SIMPLE_MLP, self).__init__(**kwargs)

		logging.info("Model name is:" + str(self.name))

		self.inputs = placeholders["features"]

		self.layer_print_ops = []
		self.num_graphs = num_graphs
		self.num_actions = 2
		self.num_nodes_per_graph = num_nodes_per_graph
		self.input_dim = input_dim
		self.num_graphs = num_graphs
		self.variable_support = variable_support

		self.placeholders = placeholders

		self.build()

	def _build_opt(self):
		
		self.fit_print_ops = []
			
		if self.method_type == "reinforce_policy":
			
			network_outputs_per_graph = tf.split(self.predict(), FLAGS.num_simultaneous_graphs)

			action_mask_per_graph = tf.split(tf.one_hot(self.placeholders['actions'], self.num_actions), FLAGS.num_simultaneous_graphs)

			neg_log_probs_per_graph = []

			for network_outputs, action_mask in zip(network_outputs_per_graph, action_mask_per_graph):

				network_probabilities = tf.nn.softmax(network_outputs)
				neg_log_prob = tf.reduce_sum(-tf.log(network_probabilities) * action_mask, axis=1)
				neg_log_probs_per_graph.append(neg_log_prob)

			rewards = self.placeholders['rewards']

			self.loss = tf.reduce_mean(neg_log_probs_per_graph * rewards)

			self.fit_print_ops.append(tf.print("loss:",self.loss))

			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.loss_grads = self.optimizer.compute_gradients(self.loss)
			self.opt_op = self.optimizer.minimize(self.loss)

		elif self.method_type == "ppo_policy":
			
			network_outputs = self.predict()

			# get old probability
			old_probabilities = self.placeholders['old_probabilities'] # one per graph
			
			network_outputs = tf.split(network_outputs, FLAGS.num_simultaneous_graphs) # ignore all zeros

			action_masks = tf.split(self.placeholders['actioned_labels_mask'], FLAGS.num_simultaneous_graphs)

			new_probabilities = []

			for outputs, action_mask in zip(network_outputs, action_masks):

				masked_logits = tf.reshape(outputs, [-1])
				probability_distribution = tf.nn.softmax(masked_logits)
				
				flat_action_mask = tf.reshape(action_mask, [-1])
			
				probability = tf.dynamic_partition(probability_distribution, flat_action_mask, 2)[1]

				new_probabilities.append(probability)

			ratios = tf.exp(tf.log(tf.add(new_probabilities,1e-9)) - tf.log(old_probabilities+1e-9)) # this is more numerically stable, I think?
			rewards = self.placeholders['rewards'] # one per transition
			
			policy_network_unclipped_loss = ratios * rewards
			policy_network_clipped_loss = tf.clip_by_value(ratios, 1.0-FLAGS.e, 1.0+FLAGS.e) * rewards
			
			minimums = tf.minimum(policy_network_unclipped_loss, policy_network_clipped_loss) # minimize the loss so we are conservative with our updates
			
			self.loss = - tf.reduce_mean(minimums) # minimizing -loss = maximizing the loss
			
			self.fit_print_ops.append(tf.print("loss:",self.loss))
		
			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

			self.loss_grads = self.optimizer.compute_gradients(self.loss)
			self.opt_op = self.optimizer.minimize(self.loss)
		
		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_predictor":
			
			network_outputs = self.predict()

			self.loss = tf.losses.mean_squared_error(network_outputs, self.placeholders['values'])
			self.fit_print_ops.append(tf.print("SV MSE ", self.loss, summarize=-1))

			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.loss_grads = self.optimizer.compute_gradients(self.loss, [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='statevalue_network') if v.name.startswith("statevalue_network/" + str(self.name))]
)
			self.opt_op = self.optimizer.minimize(self.loss)

		elif self.method_type == "rnd_target":
			pass # don't construct the optimizer for the target
		else:
			raise RuntimeError("Method type not recognised. Exiting.")

	def _build_prediction_op(self):

		self.pred_print_ops = []

		if self.method_type == "ppo_policy" or self.method_type == "reinforce_policy":
			# assuming only one graph has active nodes_mask corresponding to the actionable nodes
			masked_logits = tf.dynamic_partition(self.predict(), self.placeholders['graphs_mask'], 2)[1]
			
			#self.pred_print_ops.append(tf.print("Policy masked logits has shape ", tf.shape(masked_logits), ":", masked_logits))
			
			masked_shape = tf.shape(masked_logits)
			
			masked_logits = tf.reshape(masked_logits, [-1])
			probability_distribution = tf.nn.softmax(masked_logits)
			#self.pred_print_ops.append(tf.print("Policy direct softmax has shape ", tf.shape(probability_distribution), ":", probability_distribution))

			self.prediction_op = tf.reshape(probability_distribution, masked_shape)
			#self.masked_prediction_op = probability_distribution

		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_target" or self.method_type == "rnd_predictor":

			self.prediction_op = self.predict()

		else:
			raise RuntimeError("Method type not recognised. Exiting.")

	def _build(self):

		if self.method_type == "ppo_policy" or self.method_type == "reinforce_policy":

			# flatten and FC
			self.layers.append(Dense(input_dim=self.input_dim,
										output_dim=FLAGS.pooling_hidden,
										placeholders=self.placeholders,
										depth=0,
										model=self,
										first_pooling=True, # immediately pool, so the input it just flattened vector of the matrix [num_nodes,num_features]
										act=tf.nn.leaky_relu,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))
			
			for layer_idx in range(FLAGS.num_dense_layers-2):
				self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
											output_dim=FLAGS.pooling_hidden,
											placeholders=self.placeholders,
											depth=layer_idx+1,
											model=self,
											act=tf.nn.leaky_relu,
											dropout=False,
											sparse_inputs=False,
											logging=self.logging
											))

			# FC
			self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
										#output_dim=self.num_nodes_per_graph * self.num_actions, # flattened vector of logits corresponding to (num_nodes, num_actions), and there will be num_graphs of these vectors
										output_dim=self.num_actions, 
										placeholders=self.placeholders,
										depth=FLAGS.num_dense_layers,
										final_policy_layer=True,
										model=self,
										act=lambda x: x,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))

		elif self.method_type == "batched_statevalue" or self.method_type == "rnd_predictor" or self.method_type == "rnd_target":
			
			# flatten and FC
			self.layers.append(Dense(input_dim=self.input_dim,
										output_dim=FLAGS.pooling_hidden,
										placeholders=self.placeholders,
										depth=0,
										model=self,
										first_pooling=True,
										act=tf.nn.leaky_relu,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))
			
			for layer_idx in range(FLAGS.num_dense_layers-2):
				self.layers.append(Dense(input_dim=FLAGS.pooling_hidden,
											output_dim=FLAGS.pooling_hidden,
											placeholders=self.placeholders,
											depth=layer_idx+1,
											model=self,
											act=tf.nn.leaky_relu,
											dropout=False,
											sparse_inputs=False,
											logging=self.logging
											))

			# FC
			self.layers.append(Dense(input_dim=FLAGS.pooling_hidden, # take all the node values and reduce them to one via a dense layer
										output_dim=1,
										placeholders=self.placeholders,
										depth=FLAGS.num_dense_layers,
										model=self,
										act=lambda x: x,
										dropout=False,
										sparse_inputs=False,
										logging=self.logging
										))

	def predict(self):
		return self.outputs
