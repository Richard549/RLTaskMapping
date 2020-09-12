from gcn.models import *
from gcn.utils import *
from util import *
from agent import Action

import logging
import tensorflow as tf
from tensorflow.python.client import timeline
import scipy.sparse as sp

flags = tf.app.flags
FLAGS = flags.FLAGS

class PPONetwork():

	def __init__(self, adjacency_matrix, num_nodes, num_features, actions_vector, include_partial_solution_feature=True, sparse=False, variable_support=False, zero_non_included_nodes=False): 
		
		self.undirected_adj = adjacency_matrix
		self.sparse_undirected_adj = sp.csr_matrix(self.undirected_adj)
		self.sparse_constant_support = preprocess_adj(self.undirected_adj)

		self.include_partial_solution = include_partial_solution_feature
		self.variable_support = variable_support
		self.zero_non_included_nodes = zero_non_included_nodes

		if self.include_partial_solution == False:
			num_features -= 1

		self.num_features = num_features

		len_action_vector = len(actions_vector)
		self.actions_vector = actions_vector
		num_supports = 1

		total_num_nodes = num_nodes*FLAGS.num_simultaneous_graphs

		if self.variable_support:
			support_ph = tf.sparse_placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs*num_nodes,FLAGS.num_simultaneous_graphs*num_nodes))
		else:
			support_ph = tf.sparse_placeholder(tf.float32, shape=(num_nodes,num_nodes))

		logging.info("num nodes per graph is " + str(num_nodes))

		self.placeholders = {
			'supports': support_ph,
			'nodes_mask': tf.placeholder(tf.int32, shape=(FLAGS.num_simultaneous_graphs*num_nodes)), # these are the nodes that the computation is restricted to (during first pooling layer)
			'graphs_mask': tf.placeholder(tf.int32, shape=(FLAGS.num_simultaneous_graphs)), # these are the graphs we care about predicting the action for
			'actioned_labels_mask': tf.placeholder(tf.int32, shape=(FLAGS.num_simultaneous_graphs,len_action_vector)), # this is the actioned neuron for each graph
			'features': tf.placeholder(tf.float32, shape=(total_num_nodes, num_features)),
			'rewards': tf.placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs)),
			'old_probabilities': tf.placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs))
		}

		if FLAGS.model == 'GCN':
			self.model = GCN(self.placeholders, input_dim=num_features, num_graphs=FLAGS.num_simultaneous_graphs, num_nodes_per_graph=num_nodes, variable_support=self.variable_support, logging=True)
		elif FLAGS.model == 'MLP':
			self.model = MLP(self.placeholders, input_dim=num_features, num_graphs=FLAGS.num_simultaneous_graphs, num_nodes_per_graph=num_nodes, variable_support=self.variable_support, logging=True)
		else:
			raise RuntimeError("Requested model not recognised. Exiting.")
	
	def get_all_action_probabilities(self, sess, state, actionable_nodes, actions_vector):

		if self.include_partial_solution:
			features_per_graph = [np.copy(state.feature_matrix)]
		else:
			features_per_graph = [state.feature_matrix.transpose()[:self.num_features].transpose()]

		nodes_mask_per_graph = [actionable_nodes]
		
		if self.variable_support:
			all_nodes = np.arange(len(state.feature_matrix))
			actioned_or_actionable = np.concatenate((state.partial_solution_node_indexes, actionable_nodes))
			all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
			constrained_adj = sp.csr_matrix(self.undirected_adj)
			constrained_adj = zero_rows(self.sparse_undirected_adj, all_non_considered_nodes)
			constrained_adj = zero_columns(constrained_adj, all_non_considered_nodes)
			constrained_support = preprocess_adj(constrained_adj.tocoo())
			support_per_graph = [constrained_support]
		else:
			support_per_graph = [self.sparse_constant_support]
		
		if self.zero_non_included_nodes:
			# zero all (non-actioned or non-actionable nodes) features - we do this so that these nodes have no affect on the actioned nodes (that we care about) during convolution
			all_nodes = np.arange(len(state.feature_matrix))
			actioned_or_actionable = np.concatenate((state.partial_solution_node_indexes, actionable_nodes))
			all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
			features_per_graph[0][all_non_considered_nodes] = np.zeros(self.num_features, np.float32)
		
		feed = self.construct_masked_feed_dict(self.placeholders, features_per_graph, support_per_graph, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#run_metadata = tf.RunMetadata()

		#probabilities = sess.run([self.model.masked_prediction_op],feed_dict=feed, options=run_options, run_metadata=run_metadata)[0]
		#tl = timeline.Timeline(run_metadata.step_stats)
		#ctf = tl.generate_chrome_trace_format()
		#with open('get_best_action_timeline.json', 'w') as f:
		#	f.write(ctf)
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
			probabilities = sess.run([self.model.masked_prediction_op,self.model.pred_print_ops],feed_dict=feed)[0]
		else:
			probabilities = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0]

		return probabilities

	def get_best_action(self, sess, state, actionable_nodes, actions_vector, sample_idx=0):

		if self.include_partial_solution:
			features_per_graph = [np.copy(state.feature_matrix)]
		else:
			features_per_graph = [state.feature_matrix.transpose()[:self.num_features].transpose()]

		nodes_mask_per_graph = [1]
		
		if self.variable_support:
			all_nodes = np.arange(len(state.feature_matrix))
			actioned_or_actionable = np.concatenate((state.partial_solution_node_indexes, actionable_nodes))
			all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
			constrained_adj = sp.csr_matrix(self.undirected_adj)
			constrained_adj = zero_rows(self.sparse_undirected_adj, all_non_considered_nodes)
			constrained_adj = zero_columns(constrained_adj, all_non_considered_nodes)
			constrained_support = preprocess_adj(constrained_adj.tocoo())
			support_per_graph = [constrained_support]
		else:
			support_per_graph = [self.sparse_constant_support]
		
		if self.zero_non_included_nodes:
			# zero all (non-actioned or non-actionable nodes) features - we do this so that these nodes have no affect on the actioned nodes (that we care about) during convolution
			all_nodes = np.arange(len(state.feature_matrix))
			actioned_or_actionable = np.concatenate((state.partial_solution_node_indexes, actionable_nodes))
			all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
			features_per_graph[0][all_non_considered_nodes] = np.zeros(self.num_features, np.float32)
		
		feed = self.construct_masked_feed_dict(self.placeholders, features_per_graph, support_per_graph, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#run_metadata = tf.RunMetadata()

		#probabilities = sess.run([self.model.masked_prediction_op],feed_dict=feed, options=run_options, run_metadata=run_metadata)[0]
		#tl = timeline.Timeline(run_metadata.step_stats)
		#ctf = tl.generate_chrome_trace_format()
		#with open('get_best_action_timeline.json', 'w') as f:
		#	f.write(ctf)
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
			probabilities = sess.run([self.model.masked_prediction_op,self.model.pred_print_ops],feed_dict=feed)[0]
		else:
			probabilities = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0]

		logging.info("Probability map across actions for sample " + str(sample_idx) + " was: " + str(":".join(list(map(str,probabilities)))))

		selected_node_action = np.random.choice(range(probabilities.size), p=probabilities.ravel()) 
		node_idx, allocation = np.unravel_index(selected_node_action, probabilities.shape) 

		action = Action()
		action.node_idx = actionable_nodes[node_idx]
		action.label = allocation

		return action, probabilities[node_idx][allocation]

	def fit_to_minibatch(self, sess, features_per_transition, actions, advantages, actionable_nodes_per_transition, old_probabilities_per_transition, actioned_nodes_per_transition):
		
		if self.include_partial_solution == False:
			for graph_idx, features in enumerate(features_per_transition):
				features_per_transition[graph_idx] = features_per_transition[graph_idx].transpose()[:self.num_features].transpose()
		else:
			# not 100% sure this is necessary but just in case:
			new_features_per_transition = []
			for graph_idx, features in enumerate(features_per_transition):
				new_features_per_transition.append(np.copy(features))
			features_per_transition = new_features_per_transition

		output_dim = len(self.actions_vector)

		labels_per_graph = None
		labels_mask_per_graph = [[[action.node_idx,action.label]] for action in actions]
		
		if self.variable_support:
			support_per_transition = []
			all_nodes = np.arange(len(features_per_transition[0]))
			for transition_idx, actioned_nodes in enumerate(actioned_nodes_per_transition):
				actioned_or_actionable = np.concatenate((actioned_nodes, actionable_nodes_per_transition[transition_idx]))
				all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
				constrained_adj = zero_rows(self.sparse_undirected_adj, all_non_considered_nodes)
				constrained_adj = zero_columns(constrained_adj, all_non_considered_nodes)
				constrained_support = preprocess_adj(constrained_adj.tocoo())
				support_per_transition.append(constrained_support)
		else:
			support_per_transition = [self.sparse_constant_support]
		
		if self.zero_non_included_nodes:
			# zero all (non-actioned or non-actionable nodes) features - we do this so that these nodes have no affect on the actioned nodes (that we care about) during convolution
			all_nodes = np.arange(len(features_per_transition[graph_idx]))
			for graph_idx in range(len(features_per_transition)):
				actioned_or_actionable = np.concatenate((actioned_nodes_per_transition[graph_idx],actionable_nodes_per_transition[graph_idx]))
				all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
				features_per_transition[graph_idx][all_non_considered_nodes] = np.zeros(self.num_features, np.float32)
		
		feed = self.construct_masked_feed_dict(self.placeholders, features_per_transition, support_per_transition, FLAGS.num_simultaneous_graphs, output_dim, rewards_per_graph=advantages, labels_mask_per_graph=labels_mask_per_graph, nodes_mask_per_graph=actionable_nodes_per_transition, old_probabilities_per_transition=old_probabilities_per_transition)

		#loss, opt, grads = sess.run([self.model.loss, self.model.opt_op, self.model.loss_grads],feed_dict=feed)
		#run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG:

			logging.debug("Fitting Policy to advantages:" + str(advantages))
			loss, _, _, grads = sess.run([self.model.loss, self.model.opt_op, self.model.fit_print_ops, self.model.loss_grads],feed_dict=feed)
			logging.info("Policy gradients:" + str(grads))
		else:
			loss = sess.run([self.model.loss, self.model.opt_op],feed_dict=feed)[0]

		return loss

	def save_to_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.save(sess,filename)
	
	def load_from_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.restore(sess, filename)

	def construct_masked_feed_dict(self, placeholders, features_per_graph, support_per_graph, num_graphs, output_dim, rewards_per_graph=None, nodes_mask_per_graph=None, labels_mask_per_graph=None, old_probabilities_per_transition=None):

		feed_dict = dict()

		num_nodes = features_per_graph[0].shape[0]
		num_features = features_per_graph[0].shape[1]
		
		features = np.zeros((num_nodes*num_graphs,num_features), dtype=np.float32)
		actioned_labels_mask = np.zeros((num_graphs,output_dim), dtype=np.int32) # these are the neurons that we actioned
		nodes_mask = np.zeros((num_nodes*num_graphs), dtype=np.int32) # these are the graphs to get a decision for (predicting)
		graphs_mask = np.zeros((num_graphs), dtype=np.int32) # these are the graphs to get a decision for (predicting)
		rewards = np.zeros(num_graphs, dtype=np.float32)
		old_probabilities = np.zeros(num_graphs, dtype=np.float32)
			
		labels_masks_within_actionable = []

		for graph_idx in range(len(features_per_graph)):

			features[graph_idx*num_nodes:(graph_idx+1)*num_nodes] = features_per_graph[graph_idx]

			if rewards_per_graph is not None:
				rewards[graph_idx] = rewards_per_graph[graph_idx]
			
			if old_probabilities_per_transition is not None:
				old_probabilities[graph_idx] = old_probabilities_per_transition[graph_idx]

			if labels_mask_per_graph is not None:

				indexes = np.array(labels_mask_per_graph[graph_idx])
				indexes[:,0] += (num_nodes*graph_idx)

				for target_neuron in indexes:
					actioned_labels_mask[graph_idx, target_neuron[1]] = 1

			if nodes_mask_per_graph is not None:
				
				nodes_mask[graph_idx*num_nodes:(graph_idx+1)*num_nodes] = 1 + graph_idx
				graphs_mask[graph_idx] = graph_idx + 1

		if self.variable_support:
			supports = sp.bmat([[support_per_graph[i] if (k == i and i < len(support_per_graph))  else None for k in range(num_graphs)] for i in range(num_graphs)])
			coords = np.vstack((supports.row, supports.col)).transpose()
			values = supports.data
			dense_shape = (num_nodes*num_graphs, num_nodes*num_graphs)
			support = tf.SparseTensorValue(coords, values, dense_shape)
		else:
			supports = support_per_graph[0] # only one support
			coords = np.vstack((supports.row, supports.col)).transpose()
			values = supports.data
			dense_shape = (num_nodes, num_nodes)
			support = tf.SparseTensorValue(coords, values, dense_shape)
		
		if labels_mask_per_graph is not None:
			feed_dict.update({placeholders['actioned_labels_mask']: actioned_labels_mask})

		if nodes_mask_per_graph is not None:
			feed_dict.update({placeholders['nodes_mask']: nodes_mask})
			feed_dict.update({placeholders['graphs_mask']: graphs_mask})
		
		feed_dict.update({placeholders['features']: features})
		feed_dict.update({placeholders['rewards']: rewards})
		feed_dict.update({placeholders['old_probabilities']: old_probabilities})
		feed_dict.update({placeholders['supports']: support})


		return feed_dict
			
class ReinforcePolicyNetwork():

	def __init__(self, adjacency_matrix, num_nodes, num_features, actions_vector, include_partial_solution_feature=True, sparse=False, variable_support=False, zero_non_included_nodes=False): 
		
		self.undirected_adj = adjacency_matrix
		self.sparse_undirected_adj = sp.csr_matrix(self.undirected_adj)
		self.sparse_constant_support = preprocess_adj(self.undirected_adj)

		self.include_partial_solution = include_partial_solution_feature
		self.variable_support = variable_support
		self.zero_non_included_nodes = zero_non_included_nodes

		if self.include_partial_solution == False:
			num_features -= 1

		self.num_features = num_features

		len_action_vector = len(actions_vector)
		self.actions_vector = actions_vector
		num_supports = 1

		total_num_nodes = num_nodes*FLAGS.num_simultaneous_graphs

		if self.variable_support:
			support_ph = tf.sparse_placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs*num_nodes,FLAGS.num_simultaneous_graphs*num_nodes))
		else:
			support_ph = tf.sparse_placeholder(tf.float32, shape=(num_nodes,num_nodes))

		logging.info("num nodes per graph is " + str(num_nodes))

		self.placeholders = {
			'supports': support_ph,
			'features': tf.placeholder(tf.float32, shape=(total_num_nodes, num_features)),
			'graphs_mask': tf.placeholder(tf.int32, shape=(FLAGS.num_simultaneous_graphs)), # because we want to predict only one graph's output from a potential minibatch of graphs
			'actions': tf.placeholder(tf.int32, shape=(FLAGS.num_simultaneous_graphs)), # which neuron (from the two options) did we choose for each graphs
			'rewards': tf.placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs))
		}

		if FLAGS.model == 'GCN':
			self.model = GCN(self.placeholders, input_dim=num_features, num_graphs=FLAGS.num_simultaneous_graphs, num_nodes_per_graph=num_nodes, variable_support=self.variable_support, logging=True)
		elif FLAGS.model == 'MLP':
			self.model = MLP(self.placeholders, input_dim=num_features, num_graphs=FLAGS.num_simultaneous_graphs, num_nodes_per_graph=num_nodes, variable_support=self.variable_support, logging=True)
		else:
			raise RuntimeError("Requested model not recognised. Exiting.")
	
	def get_best_action(self, sess, state, actionable_nodes, actions_vector, sample_idx=0):

		if self.include_partial_solution:
			features_per_graph = [np.copy(state.feature_matrix)]
		else:
			features_per_graph = [state.feature_matrix.transpose()[:self.num_features].transpose()]

		nodes_mask_per_graph = [1]
		
		if self.variable_support:
			all_nodes = np.arange(len(state.feature_matrix))
			actioned_or_actionable = np.concatenate((state.partial_solution_node_indexes, actionable_nodes))
			all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
			constrained_adj = sp.csr_matrix(self.undirected_adj)
			constrained_adj = zero_rows(self.sparse_undirected_adj, all_non_considered_nodes)
			constrained_adj = zero_columns(constrained_adj, all_non_considered_nodes)
			constrained_support = preprocess_adj(constrained_adj.tocoo())
			support_per_graph = [constrained_support]
		else:
			support_per_graph = [self.sparse_constant_support]
		
		if self.zero_non_included_nodes:
			# zero all (non-actioned or non-actionable nodes) features - we do this so that these nodes have no affect on the actioned nodes (that we care about) during convolution
			all_nodes = np.arange(len(state.feature_matrix))
			actioned_or_actionable = np.concatenate((state.partial_solution_node_indexes, actionable_nodes))
			all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
			features_per_graph[0][all_non_considered_nodes] = np.zeros(self.num_features, np.float32)
		
		feed = self.construct_masked_feed_dict(self.placeholders, features_per_graph, support_per_graph, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#run_metadata = tf.RunMetadata()

		#probabilities = sess.run([self.model.masked_prediction_op],feed_dict=feed, options=run_options, run_metadata=run_metadata)[0]
		#tl = timeline.Timeline(run_metadata.step_stats)
		#ctf = tl.generate_chrome_trace_format()
		#with open('get_best_action_timeline.json', 'w') as f:
		#	f.write(ctf)
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
			probabilities = sess.run([self.model.prediction_op,self.model.pred_print_ops],feed_dict=feed)[0]
		else:
			probabilities = sess.run([self.model.prediction_op],feed_dict=feed)[0]

		logging.info("Probability map across actions for sample " + str(sample_idx) + " was: " + str(":".join(list(map(str,probabilities)))))

		selected_node_action = np.random.choice(range(probabilities.size), p=probabilities.ravel()) 
		node_idx, allocation = np.unravel_index(selected_node_action, probabilities.shape) 

		action = Action()
		action.node_idx = actionable_nodes[node_idx]
		action.label = allocation

		return action, probabilities[node_idx][allocation]

	def fit_to_minibatch(self, sess, features_per_transition, actions, advantages, actionable_nodes_per_transition, old_probabilities_per_transition, actioned_nodes_per_transition=None):
		
		if self.include_partial_solution == False:
			for graph_idx, features in enumerate(features_per_transition):
				features_per_transition[graph_idx] = features_per_transition[graph_idx].transpose()[:self.num_features].transpose()
		else:
			# not 100% sure this is necessary but just in case:
			new_features_per_transition = []
			for graph_idx, features in enumerate(features_per_transition):
				new_features_per_transition.append(np.copy(features))
			features_per_transition = new_features_per_transition

		output_dim = len(self.actions_vector)

		labels_per_graph = None
		actions_per_graph = [[[action.node_idx,action.label]] for action in actions]
		
		if self.variable_support:
			support_per_transition = []
			all_nodes = np.arange(len(features_per_transition[0]))
			for transition_idx, actioned_nodes in enumerate(actioned_nodes_per_transition):
				actioned_or_actionable = np.concatenate((actioned_nodes, actionable_nodes_per_transition[transition_idx]))
				all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
				constrained_adj = zero_rows(self.sparse_undirected_adj, all_non_considered_nodes)
				constrained_adj = zero_columns(constrained_adj, all_non_considered_nodes)
				constrained_support = preprocess_adj(constrained_adj.tocoo())
				support_per_transition.append(constrained_support)
		else:
			support_per_transition = [self.sparse_constant_support]
		
		if self.zero_non_included_nodes:
			# zero all (non-actioned or non-actionable nodes) features - we do this so that these nodes have no affect on the actioned nodes (that we care about) during convolution
			all_nodes = np.arange(len(features_per_transition[graph_idx]))
			for graph_idx in range(len(features_per_transition)):
				actioned_or_actionable = np.concatenate((actioned_nodes_per_transition[graph_idx],actionable_nodes_per_transition[graph_idx]))
				all_non_considered_nodes = np.setxor1d(all_nodes, actioned_or_actionable)
				features_per_transition[graph_idx][all_non_considered_nodes] = np.zeros(self.num_features, np.float32)
		
		feed = self.construct_masked_feed_dict(self.placeholders, features_per_transition, support_per_transition, FLAGS.num_simultaneous_graphs, output_dim, rewards_per_graph=advantages, actions=actions_per_graph, nodes_mask_per_graph=actionable_nodes_per_transition, old_probabilities_per_transition=old_probabilities_per_transition)

		#loss, opt, grads = sess.run([self.model.loss, self.model.opt_op, self.model.loss_grads],feed_dict=feed)
		#run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG:

			logging.debug("Fitting Policy to advantages:" + str(advantages))
			loss, _, _, grads = sess.run([self.model.loss, self.model.opt_op, self.model.fit_print_ops, self.model.loss_grads],feed_dict=feed)
			#logging.info("Policy gradients:" + str(grads))
		else:
			loss = sess.run([self.model.loss, self.model.opt_op],feed_dict=feed)[0]

		return loss

	def save_to_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.save(sess,filename)
	
	def load_from_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.restore(sess, filename)

	def construct_masked_feed_dict(self, placeholders, features_per_graph, support_per_graph, num_graphs, output_dim, rewards_per_graph=None, nodes_mask_per_graph=None, actions=None, old_probabilities_per_transition=None):

		feed_dict = dict()

		num_nodes = features_per_graph[0].shape[0]
		num_features = features_per_graph[0].shape[1]
		
		features = np.zeros((num_nodes*num_graphs,num_features), dtype=np.float32)
		actioned_labels_per_graph = np.zeros(num_graphs, dtype=np.int32)
		nodes_mask = np.zeros((num_nodes*num_graphs), dtype=np.int32) # these are the graphs to get a decision for (predicting)
		graphs_mask = np.zeros((num_graphs), dtype=np.int32) # these are the graphs to get a decision for (predicting)
		rewards = np.zeros(num_graphs, dtype=np.float32)
		old_probabilities = np.zeros(num_graphs, dtype=np.float32)
			
		labels_masks_within_actionable = []

		for graph_idx in range(len(features_per_graph)):

			features[graph_idx*num_nodes:(graph_idx+1)*num_nodes] = features_per_graph[graph_idx]

			if rewards_per_graph is not None:
				rewards[graph_idx] = rewards_per_graph[graph_idx]
			
			if old_probabilities_per_transition is not None:
				old_probabilities[graph_idx] = old_probabilities_per_transition[graph_idx]

			if actions is not None:

				for action in actions[graph_idx]:
					actioned_labels_per_graph[graph_idx] = action[1]

			if nodes_mask_per_graph is not None:
				
				nodes_mask[graph_idx*num_nodes:(graph_idx+1)*num_nodes] = 1 + graph_idx
				graphs_mask[graph_idx] = graph_idx + 1

		if self.variable_support:
			supports = sp.bmat([[support_per_graph[i] if (k == i and i < len(support_per_graph))  else None for k in range(num_graphs)] for i in range(num_graphs)])
			coords = np.vstack((supports.row, supports.col)).transpose()
			values = supports.data
			dense_shape = (num_nodes*num_graphs, num_nodes*num_graphs)
			support = tf.SparseTensorValue(coords, values, dense_shape)
		else:
			supports = support_per_graph[0] # only one support
			coords = np.vstack((supports.row, supports.col)).transpose()
			values = supports.data
			dense_shape = (num_nodes, num_nodes)
			support = tf.SparseTensorValue(coords, values, dense_shape)
		
		if actions is not None:
			feed_dict.update({placeholders['actions']: actioned_labels_per_graph})

		if nodes_mask_per_graph is not None:
			feed_dict.update({placeholders['graphs_mask']: graphs_mask})
		
		feed_dict.update({placeholders['features']: features})
		feed_dict.update({placeholders['rewards']: rewards})
		feed_dict.update({placeholders['supports']: support})


		return feed_dict
			
