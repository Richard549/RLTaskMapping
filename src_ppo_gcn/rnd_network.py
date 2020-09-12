from gcn.models import *
from gcn.utils import *
from util import *
from agent import Action

import logging
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class RNDNetwork():

	def __init__(self, adjacency_matrix, num_nodes, num_features, actions_vector, include_partial_solution_feature=True): 

		self.undirected_adj = adjacency_matrix
		#self.support = preprocess_adj(adjacency_matrix).todense()

		self.output_dim = len(actions_vector) # TODO just the size of the actions space? I think the paper considers this arbitrary
		num_supports = 1
		self.action_vector = actions_vector
		self.include_partial_solution = include_partial_solution_feature

		if self.include_partial_solution == False:
			num_features -= 1

		self.num_features = num_features

		total_num_nodes = num_nodes*FLAGS.num_simultaneous_graphs
		self.placeholders = {
			'supports': tf.placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs,num_nodes,num_nodes)),
			'features': tf.placeholder(tf.float32, shape=(total_num_nodes, num_features)),
			'target_network_values': tf.placeholder(tf.float32, shape=(FLAGS.num_simultaneous_graphs)), # one per graph
			'nodes_mask': tf.placeholder(tf.float32, shape=(total_num_nodes,self.output_dim)) # these are the outputs neurons of the actioned nodes for the particular graph
		}

		if FLAGS.model == 'GCN':
			self.model = GCN(self.placeholders, input_dim=num_features, logging=True)
		elif FLAGS.model == 'MLP':
			self.model = MLP(self.placeholders, input_dim=num_features, logging=True)
		else:
			raise RuntimeError("Requested model not recognised. Exiting.")

	def get_value_for_state(self, sess, features, actioned_nodes):

		features_per_graph = [features]
		if self.include_partial_solution:
			features_per_graph = [features]
		else:
			features_per_graph = [features.transpose()[:self.num_features].transpose()]

		nodes_per_graph = [actioned_nodes]

		support_per_graph = []

		# the support for each graph is the undirected adj but all non-actioned nodes are 0s, then preprocessed
		constrained_adj = np.copy(self.undirected_adj)
		all_nodes = list(range(len(features)))
		all_non_actioned_nodes = np.setdiff1d(all_nodes, actioned_nodes)
		constrained_adj[all_non_actioned_nodes] = np.zeros(len(all_nodes)) # all non-actioned nodes have no input dependencies
		constrained_adj.transpose()[all_non_actioned_nodes] = np.zeros(len(all_nodes)) # all non-actioned nodes have no output dependencies
		constrained_support = preprocess_adj(constrained_adj).todense()

		support_per_graph = [constrained_support]
		
		feed = self.construct_masked_feed_dict(self.placeholders, features_per_graph, support_per_graph, FLAGS.num_simultaneous_graphs, self.output_dim, nodes_per_graph)

		value_for_state = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0]

		return value_for_state

	def fit_to_minibatch(self, sess, features_per_graph, values_per_graph, actioned_nodes_per_graph):
	
		if self.include_partial_solution == False:
			for graph_idx, features in enumerate(features_per_graph):
				features_per_graph[graph_idx] = features_per_graph[graph_idx].transpose()[:self.num_features].transpose()
		
		support_per_graph = []

		all_nodes = list(range(len(features_per_graph[0])))
		for actioned_nodes in actioned_nodes_per_graph:
			# the support for each graph is the undirected adj but all non-actioned nodes are 0s, then preprocessed
			constrained_adj = np.copy(self.undirected_adj)
			all_non_actioned_nodes = np.setdiff1d(all_nodes, actioned_nodes)
			constrained_adj[all_non_actioned_nodes] = np.zeros(len(all_nodes)) # all non-actioned nodes have no input dependencies
			constrained_adj.transpose()[all_non_actioned_nodes] = np.zeros(len(all_nodes)) # all non-actioned nodes have no output dependencies
			constrained_support = preprocess_adj(constrained_adj).todense()
			support_per_graph.append(constrained_support)
	
		feed = self.construct_masked_feed_dict(self.placeholders, features_per_graph, support_per_graph, FLAGS.num_simultaneous_graphs, self.output_dim, actioned_nodes_per_graph, values_per_graph)

		loss, opt = sess.run([self.model.loss, self.model.opt_op],feed_dict=feed)

		return loss
	
	def save_to_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.save(sess, filename)

	def load_from_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.restore(sess, filename)

	def construct_masked_feed_dict(self, placeholders, features_per_graph, support_per_graph, num_graphs, output_dim, nodes_per_graph, values_per_graph=None):
	
		feed_dict = dict()

		num_nodes = features_per_graph[0].shape[0]
		num_features = features_per_graph[0].shape[1]
		
		features = np.zeros((num_nodes*num_graphs,num_features), dtype=np.float32)
		nodes_mask = np.zeros((num_nodes*num_graphs,2), dtype=np.float32) # these are the actioned (partial solution) nodes
		values = np.zeros(num_graphs, dtype=np.float32) # only used if training
		supports = np.zeros((num_graphs, num_nodes, num_nodes), dtype=np.float32)

		for graph_idx in range(len(features_per_graph)):

			features[graph_idx*num_nodes:(graph_idx+1)*num_nodes] = features_per_graph[graph_idx]

			if values_per_graph is not None:
				values[graph_idx] = values_per_graph[graph_idx]

			if nodes_per_graph is not None:
				indexes = np.array(nodes_per_graph[graph_idx]) + (num_nodes*graph_idx)
				nodes_mask[indexes] = np.ones(2)
			
			supports[graph_idx] = support_per_graph[graph_idx]

		feed_dict.update({placeholders['nodes_mask']: nodes_mask})
		feed_dict.update({placeholders['features']: features})
		feed_dict.update({placeholders['supports']: supports})
		feed_dict.update({placeholders['target_network_values']: values})

		return feed_dict
