from gcn.models import *
from gcn.utils import *
from util import *
from agent import Action

import logging
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.python.client import timeline

flags = tf.app.flags
FLAGS = flags.FLAGS

class BatchedStateValueNetwork():

	def __init__(self, adjacency_matrix, num_nodes, num_features, actions_vector, include_partial_solution_feature=True, variable_support=False, zero_non_included_nodes=False): 

		self.undirected_adj = adjacency_matrix
		self.sparse_undirected_adj = sp.csr_matrix(self.undirected_adj)
		self.sparse_constant_support = preprocess_adj(self.undirected_adj)

		logging.debug("Dense support is:\n" + str(self.sparse_constant_support.todense()))

		self.output_dim = 1
		num_supports = 1
		self.action_vector = actions_vector
		self.include_partial_solution = include_partial_solution_feature

		if self.include_partial_solution == False:
			num_features -= 1

		self.num_features = num_features
		self.variable_support = variable_support
		self.zero_non_included_nodes = zero_non_included_nodes
		
		if self.variable_support:
			support_ph = tf.sparse_placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs*num_nodes,FLAGS.num_simultaneous_graphs*num_nodes))
		else:
			support_ph = tf.sparse_placeholder(tf.float32, shape=(num_nodes,num_nodes))

		self.placeholders = {
			'supports': support_ph,
			'features': tf.placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs*num_nodes, num_features)),
			'values': tf.placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs)),
			'nodes_mask': tf.placeholder(tf.int32, shape=(FLAGS.num_simultaneous_graphs*num_nodes)),
			#'nodes_mask_to_split': tf.placeholder(tf.int32, shape=(FLAGS.num_simultaneous_graphs*num_nodes)), # this is exactly the same as nodes_mask except we only use 1s, to allow dynamic partition
		}

		if FLAGS.model == 'GCN':
			self.model = GCN(self.placeholders, input_dim=num_features, num_graphs=FLAGS.num_simultaneous_graphs, num_nodes_per_graph=num_nodes, variable_support=self.variable_support, logging=True)
		elif FLAGS.model == 'MLP':
			self.model = MLP(self.placeholders, input_dim=num_features, num_graphs=FLAGS.num_simultaneous_graphs, num_nodes_per_graph=num_nodes, variable_support=self.variable_support, logging=True)
		else:
			raise RuntimeError("Requested model not recognised. Exiting.")

	def get_value_for_state(self, sess, features, actioned_nodes_at_state):
		
		if self.include_partial_solution:
			features_per_graph = [np.copy(features)]
		else:
			features_per_graph = [features.transpose()[:self.num_features].transpose()]

		nodes_per_graph = [actioned_nodes_at_state]
		
		if self.variable_support:
			all_nodes = np.arange(len(features))
			all_non_actioned_nodes = np.setxor1d(all_nodes, actioned_nodes_at_state)

			constrained_adj = sp.csr_matrix(self.undirected_adj)
			constrained_adj = zero_rows(self.sparse_undirected_adj, all_non_actioned_nodes)
			constrained_adj = zero_columns(constrained_adj, all_non_actioned_nodes)
			constrained_support = preprocess_adj(constrained_adj.tocoo())

			support_per_graph = [constrained_support]
		else:
			support_per_graph = [self.sparse_constant_support]

		if self.zero_non_included_nodes:
			# zero all non-actioned nodes features - we do this so that these nodes have no affect on the actioned nodes (that we care about) during convolution
			all_nodes = np.arange(len(features))
			all_non_actioned_nodes = np.setxor1d(all_nodes, actioned_nodes_at_state)
			features_per_graph[0][all_non_actioned_nodes] = np.zeros(self.num_features, np.float32)

		feed = self.construct_masked_feed_dict(self.placeholders, features_per_graph, support_per_graph, FLAGS.num_simultaneous_graphs, self.output_dim, nodes_per_graph)

		#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#run_metadata = tf.RunMetadata()
		#value_for_state = sess.run([self.model.masked_prediction_op],feed_dict=feed, options=run_options, run_metadata=run_metadata)[0]
		#tl = timeline.Timeline(run_metadata.step_stats)
		#ctf = tl.generate_chrome_trace_format()
		#with open('get_value_for_state_timeline.json', 'w') as f:
		#	f.write(ctf)
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
			value_for_state = sess.run([self.model.masked_prediction_op,self.model.pred_print_ops],feed_dict=feed)[0][0]
		else:
			value_for_state = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0][0]

		#sess.run([self.model.layer_print_ops],feed_dict=feed)
		#value_for_state = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0][0]
		
		logging.debug("Value for state:" + str(value_for_state))
		return value_for_state

	def fit_to_minibatch(self, sess, features_per_graph, actioned_nodes_per_graph, values):
	
		#logging.info("Feeding values:" + str(values))
		if self.include_partial_solution == False:
			for graph_idx, features in enumerate(features_per_graph):
				features_per_graph[graph_idx] = features_per_graph[graph_idx].transpose()[:self.num_features].transpose()
		else:
			# not 100% sure this is necessary but just in case:
			new_features_per_graph = []
			for graph_idx, features in enumerate(features_per_graph):
				new_features_per_graph.append(np.copy(features))
			features_per_graph = new_features_per_graph

		if self.variable_support:
			support_per_graph = []

			all_nodes = np.arange(len(features_per_graph[0]))
			for actioned_nodes in actioned_nodes_per_graph:
				constrained_adj = sp.csr_matrix(self.undirected_adj)
				all_non_actioned_nodes = np.setxor1d(all_nodes, actioned_nodes)
				#constrained_adj = delete_from_csr(self.sparse_undirected_adj, all_non_actioned_nodes, all_non_actioned_nodes)
				constrained_adj = zero_rows(self.sparse_undirected_adj, all_non_actioned_nodes)
				constrained_adj = zero_columns(constrained_adj, all_non_actioned_nodes)
				constrained_support = preprocess_adj(constrained_adj.tocoo())
				support_per_graph.append(constrained_support)

		else:
			support_per_graph = [self.sparse_constant_support]

		if self.zero_non_included_nodes:
			all_nodes = np.arange(len(features_per_graph[0]))
			for graph_idx, actioned_nodes in enumerate(actioned_nodes_per_graph):
				all_non_actioned_nodes = np.setxor1d(all_nodes, actioned_nodes)
				features_per_graph[graph_idx][all_non_actioned_nodes] = np.zeros(self.num_features, np.float32)

		feed = self.construct_masked_feed_dict(self.placeholders, features_per_graph, support_per_graph, FLAGS.num_simultaneous_graphs, self.output_dim, actioned_nodes_per_graph, values=values)

		##logging.info(self.model.method_type)
		#logging.info(self.model.loss_grads)
		#loss, opt, _, _ = sess.run([self.model.loss, self.model.opt_op, self.model.print_ops, self.model.print_op],feed_dict=feed)
		#loss, opt, grads = sess.run([self.model.loss, self.model.opt_op, self.model.loss_grads],feed_dict=feed)
		#grads = sess.run([grad for grad, _ in self.model.loss_grads])
		#logging.info("grads")

		'''
		variables_names = [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='statevalue_network') if v.name.startswith("statevalue_network/gcn")]
		values = sess.run(variables_names)
		for k, v in zip(variables_names, values):
			pass
			#logging.info("Value of " + str(k) + ":" + str(v))
		'''

		#loss, opt = sess.run([self.model.loss, self.model.opt_op],feed_dict=feed)
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
			logging.debug("Fitting SV network to true values:" + str(values))
			#logging.debug("The gradient operation is:" + str(self.model.loss_grads))
			#loss, _, _, grads = sess.run([self.model.loss, self.model.opt_op, self.model.fit_print_ops, self.model.loss_grads],feed_dict=feed)
			loss = sess.run([self.model.loss, self.model.opt_op, self.model.fit_print_ops],feed_dict=feed)[0]

			'''
			grad_idx = 0
			for k, v in zip(variables_names, values):
				logging.info("Gradient of " + str(variables_names[grad_idx]) + ":" + str(grads[grad_idx]))
				grad_idx += 1
			'''

		else:
			loss = sess.run([self.model.loss, self.model.opt_op],feed_dict=feed)[0]
			#logging.info("State value gradients:" + str(grads))
		#logging.info("success")
		#exit(0)
		
		'''
		values = sess.run(variables_names)
		for k, v in zip(variables_names, values):
			pass
			#logging.info("New value of " + str(k) + ":" + str(v))
		'''

		#exit(0)

		return loss
	
	def save_to_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.save(sess, filename)

	def load_from_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.restore(sess, filename)

	def construct_masked_feed_dict(self, placeholders, features_per_graph, support_per_graph, num_graphs, output_dim, nodes_per_graph, values=None):

		feed_dict = dict()

		num_nodes = features_per_graph[0].shape[0]
		num_features = features_per_graph[0].shape[1]

		features = np.zeros((num_nodes*num_graphs,num_features), dtype=np.float32)
		values_ph = np.zeros(num_graphs, dtype=np.float32)
		nodes_mask = np.zeros((num_nodes*num_graphs), dtype=np.int32)
		#single_nodes_mask = np.zeros((num_nodes*num_graphs,1), dtype=np.float32)

		for graph_idx in range(len(features_per_graph)):

			features[graph_idx*num_nodes:(graph_idx+1)*num_nodes] = features_per_graph[graph_idx]

			if values is not None:
				values_ph[graph_idx] = values[graph_idx]
			
			indexes = np.array(nodes_per_graph[graph_idx]) + (num_nodes*graph_idx)
			nodes_mask[indexes] = 1+graph_idx

		#single_nodes_mask[nodes_per_graph[graph_idx]] = 1

		#single_nodes_mask = sp.coo_matrix(single_nodes_mask)
		#coords = np.vstack((single_nodes_mask.row, single_nodes_mask.col)).transpose()
		#values = single_nodes_mask.data
		#dense_shape = (num_nodes*num_graphs)
		#single_nodes_mask = tf.SparseTensorValue(coords, values, dense_shape)

		if self.variable_support:
			supports = sp.bmat([[support_per_graph[i] if (k == i and i < len(support_per_graph))  else None for k in range(num_graphs)] for i in range(num_graphs)])
			coords = np.vstack((supports.row, supports.col)).transpose()
			values = supports.data
			dense_shape = (num_nodes*num_graphs, num_nodes*num_graphs)
			support = tf.SparseTensorValue(coords, values, dense_shape)

			#logging.info("Variable support given:\n" + str(supports.todense()))

		else:
			supports = support_per_graph[0] # only one support
			coords = np.vstack((supports.row, supports.col)).transpose()
			values = supports.data
			dense_shape = (num_nodes, num_nodes)
			support = tf.SparseTensorValue(coords, values, dense_shape)

		feed_dict.update({placeholders['values']: values_ph})
		feed_dict.update({placeholders['nodes_mask']: nodes_mask})
		feed_dict.update({placeholders['features']: features})
		feed_dict.update({placeholders['supports']: support})

		return feed_dict

