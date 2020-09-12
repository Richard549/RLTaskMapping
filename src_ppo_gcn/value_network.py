from gcn.models import *
from gcn.utils import *
from util import *
from agent import Action

import logging
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def construct_masked_feed_dict(placeholders, features_per_graph, support, num_graphs, output_dim, nodes_mask_per_graph=None, labels_per_graph=None, labels_mask_per_graph=None, current_labels=None):
	# labels_mask_per_graph[i] should be a list of [node index, action_index] that we want to constrain the computation to, for graph i
	# labels_per_graph[i] should be a list of corresponding labels for each target label in label_mask_per_graph[i], for graph i
	# nodes_mask_per_graph[i] is list of desired nodes indexes for graph i

	feed_dict = dict()

	#logging.debug("There are " + str(len(features_per_graph)) + " features per graph")
	num_nodes = features_per_graph[0].shape[0]
	num_features = features_per_graph[0].shape[1]
	
	features = np.zeros((num_nodes*num_graphs,num_features), dtype=np.float32)
	labels = np.zeros((num_nodes*num_graphs,output_dim), dtype=np.float32)
	if current_labels is not None:
		logging.debug("current labels are:" + str(current_labels))
		labels[:] = current_labels
	labels_mask = np.zeros((num_nodes*num_graphs,output_dim), dtype=np.int32)
	nodes_mask = np.zeros((num_nodes*num_graphs), dtype=np.int32)

	if nodes_mask_per_graph is None and labels_mask_per_graph is None:
		raise RuntimeError("Must provide a mask for feeding to TensorFlow.")

	for graph_idx in range(len(features_per_graph)):

		features[graph_idx*num_nodes:(graph_idx+1)*num_nodes] = features_per_graph[graph_idx]

		if labels_per_graph is not None:

			indexes = np.array(labels_mask_per_graph[graph_idx])
			indexes[:,0] += (num_nodes*graph_idx)

			#logging.debug("Shape of indexes:" + str(indexes.shape))

			#logging.debug("Shape of labels: " + str(labels.shape))
			#logging.debug("Labels before are " + str(labels))
			#logging.debug("Label values are: " + str(labels_per_graph[graph_idx]))
			labels[tuple(indexes.T)] = labels_per_graph[graph_idx]
			labels_mask[tuple(indexes.T)] = 1 # TODO this is incorrect?
			#logging.debug("The indexes are: " + str(indexes))
			#logging.debug("The tuple indexes are: " + str(tuple(indexes)))
			#logging.debug("Labels after are " + str(labels))

		if nodes_mask_per_graph is not None:
			
			indexes = np.array(nodes_mask_per_graph[graph_idx]) + (num_nodes*graph_idx)

			nodes_mask[indexes] = 1

	# the result is:
	# labels[i] shape is [num_nodes, num_actions]
	# labels_mask[i] is binary mask of shape [num_nodes, num_actions], 1 meaning the output neuron is computed, 0 meaning the neuron is discarded 
	# nodes_mask[i] is a binary mask of shape [num_nodes], 1 meaning return the output neurons for the neuron, 0 meaning don't return

	if labels_per_graph is not None:
		feed_dict.update({placeholders['labels']: labels})
		feed_dict.update({placeholders['labels_mask']: labels_mask})

	if nodes_mask_per_graph is not None:
		feed_dict.update({placeholders['nodes_mask']: nodes_mask})
	
	feed_dict.update({placeholders['features']: features})
	feed_dict.update({placeholders['support']: support})

	return feed_dict

class ValueNetwork():

	def __init__(self, adjacency_matrix, num_nodes, num_features, action_vector): 

		self.adj = adjacency_matrix
		self.support = preprocess_adj(adjacency_matrix).todense()

		len_action_vector = len(action_vector)
		num_supports = 1
		self.action_vector = action_vector

		self.method_type = FLAGS.method_type
		logging.debug("Method type for value network: " + str(self.method_type))

		total_num_nodes = num_nodes*FLAGS.num_simultaneous_graphs
		self.placeholders = {
			'support': tf.placeholder(tf.float32, shape=(num_nodes,num_nodes)),
			'labels': tf.placeholder_with_default(np.zeros((total_num_nodes,len_action_vector),dtype=np.float32), shape=(total_num_nodes,len_action_vector)),
			'labels_mask': tf.placeholder_with_default(np.zeros((total_num_nodes,len_action_vector),dtype=np.int32), shape=(total_num_nodes,len_action_vector)),
			'nodes_mask': tf.placeholder_with_default(np.zeros(total_num_nodes,dtype=np.int32), shape=(total_num_nodes)),
			'features': tf.placeholder(tf.float32, shape=(total_num_nodes, num_features))
		}

		if FLAGS.model == 'GCN':
			self.model = GCN(self.placeholders, input_dim=num_features, logging=True)
		elif FLAGS.model == 'MLP':
			self.model = MLP(self.placeholders, input_dim=num_features, logging=True)
		else:
			raise RuntimeError("Requested model not recognised. Exiting.")

		self.transition_minibatch = []
	
	def get_reward_from_action(self, sess, state, action, actions_vector):

		features_per_graph = [state.feature_matrix]
		nodes_mask_per_graph = [[action.node_idx]]

		feed = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		rewards_for_node = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0][0]

		reward = rewards_for_node[action.label]

		return reward

	def get_all_rewards(self, sess, state, actionable_nodes, actions_vector):
		
		features_per_graph = [state.feature_matrix]
		#nodes_mask_per_graph = [[node for node in actionable_nodes]]
		nodes_mask_per_graph = [[node for node in range(state.feature_matrix.shape[0])]]

		feed = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		all_rewards = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0]

		return all_rewards
	
	def get_average_reward_across_all_actions(self, sess, state, actionable_nodes, actions_vector):

		features_per_graph = [state.feature_matrix]
		nodes_mask_per_graph = [[actionable_nodes]]
	
		logging.debug("Getting average reward across all actions at state")	
		feed = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		average_reward = sess.run([self.model.average_reward_op],feed_dict=feed)[0]
		return average_reward
	
	def get_total_reward_summed_across_all_actions(self, sess, state, actionable_nodes, actions_vector):

		features_per_graph = [state.feature_matrix]
		nodes_mask_per_graph = [[actionable_nodes]]
	
		logging.debug("Getting total summed reward across all actions at state")
		feed = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		total_reward = sess.run([self.model.total_reward_op],feed_dict=feed)[0]
		return total_reward

	def get_best_actions(self, sess, state_per_transition, actionable_nodes_per_transition, actions_vector):

		logging.debug("There are " + str(len(state_per_transition)) + " states per transitions.")

		features_per_graph = [state.feature_matrix for state in state_per_transition]
		nodes_mask_per_graph = [actionable_node_indexes for actionable_node_indexes in actionable_nodes_per_transition]

		logging.debug("Getting best actions")	
		feed = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(actions_vector), nodes_mask_per_graph=nodes_mask_per_graph)

		rewards = sess.run([self.model.masked_prediction_op],feed_dict=feed)[0]
		
		best_future_action_per_transition = []
		best_future_reward_per_transition = []

		# TODO should be general to output_dim, not just cpu/gpu
		num_nodes_analyzed = 0
		for graph_idx in range(len(state_per_transition)):

			rewards_for_transition = rewards[num_nodes_analyzed:num_nodes_analyzed+len(actionable_nodes_per_transition[graph_idx])]

			# length of this vector == actionable_nodes_per_transition[graph_idx]
			cpu_rewards = rewards_for_transition.transpose()[0].transpose()
			gpu_rewards = rewards_for_transition.transpose()[1].transpose()

			max_cpu_reward = np.amax(cpu_rewards)
			max_gpu_reward = np.amax(gpu_rewards)

			best_cpu_index = np.random.choice(np.argwhere(cpu_rewards == max_cpu_reward).flatten(),1)[0]
			best_gpu_index = np.random.choice(np.argwhere(gpu_rewards == max_gpu_reward).flatten(),1)[0]

			action = Action()

			if max_cpu_reward == max_gpu_reward:

				allocation = np.random.choice(actions_vector,1)[0]
				
				action.node_idx = [actionable_nodes_per_transition[graph_idx][best_cpu_index], actionable_nodes_per_transition[graph_idx][best_gpu_index]][allocation]
				action.label = allocation
				best_reward = max_cpu_reward

			elif max_cpu_reward > max_gpu_reward:

				action.node_idx = actionable_nodes_per_transition[graph_idx][best_cpu_index]
				action.label = 0
				best_reward = max_cpu_reward

			elif max_cpu_reward < max_gpu_reward:

				action.node_idx = actionable_nodes_per_transition[graph_idx][best_gpu_index]
				action.label = 1
				best_reward = max_gpu_reward

			best_future_action_per_transition.append(action)
			best_future_reward_per_transition.append(best_reward)

			num_nodes_analyzed += len(actionable_nodes_per_transition[graph_idx])

		return best_future_action_per_transition, best_future_reward_per_transition

	def add_to_minibatch(self, state, action, cumulative_reward_from_action):
		self.transition_minibatch.append([state, action, cumulative_reward_from_action])

	def fit_to_minibatch(self, sess):
	
		features_per_graph = []
		labels_per_graph = []
		labels_mask_per_graph = []
		actions_mask_per_graph = []

		nodes_mask_per_graph = []
		true_rewards = []

		for transition in self.transition_minibatch:
		
			features = transition[0].feature_matrix
			action = transition[1]
			reward = transition[2]

			labels_targets = [[action.node_idx, action.label]] # fit only to the loss for one neuron for this graph (but do this across all graphs of the minibatch)
			labels = [reward] # fit the neuron value to this

			logging.debug("Label targets are " + str(labels_targets))

			features_per_graph.append(features)
			labels_per_graph.append(labels)
			labels_mask_per_graph.append(labels_targets)

			nodes_mask_per_graph.append([action.node_idx])
			true_rewards.append(reward)
		
		# TEMP
		#all_nodes_mask = [np.array(range(features_per_graph[0].shape[0])) for graph_idx in features_per_graph]
		#feed2 = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(self.action_vector), nodes_mask_per_graph=all_nodes_mask)
		#all_current_rewards = sess.run([self.model.masked_prediction_op],feed_dict=feed2)[0]

		if logging.getLogger().getEffectiveLevel() == logging.DEBUG and False:
			feed2 = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(self.action_vector), nodes_mask_per_graph=nodes_mask_per_graph)
			rewards = sess.run([self.model.masked_prediction_op],feed_dict=feed2)[0]
			logging.debug("Current rewards:" + str(rewards))
			logging.debug("Fitting to the true rewards: " + str(np.array(true_rewards)))

		feed = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(self.action_vector), labels_per_graph=labels_per_graph, labels_mask_per_graph=labels_mask_per_graph)
		#feed = construct_masked_feed_dict(self.placeholders, features_per_graph, self.support, FLAGS.num_simultaneous_graphs, len(self.action_vector), labels_per_graph=labels_per_graph, labels_mask_per_graph=labels_mask_per_graph, current_labels=all_current_rewards)

		#logging.debug("The fitting feed is:")
		#logging.debug("labels_mask:" + str(feed[self.placeholders['labels_mask']]))
		#logging.debug("labels:" + str(feed[self.placeholders['labels']]))

		if logging.getLogger().getEffectiveLevel() == logging.DEBUG and False:
			loss, opt, print_op = sess.run([self.model.loss, self.model.opt_op, self.model.print_op],feed_dict=feed)
		else:
			loss, opt = sess.run([self.model.loss, self.model.opt_op],feed_dict=feed)
		
		if logging.getLogger().getEffectiveLevel() == logging.DEBUG and False:
			rewards = sess.run([self.model.masked_prediction_op],feed_dict=feed2)[0]
			logging.debug("New rewards:" + str(rewards))

		# empty the feed dict
		self.transition_minibatch.clear()

		return loss

	def become_copy_of(self, sess, other_qnetwork, other_scope="continuous_qnetwork", my_scope="target_qnetwork"):
		# Taken from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py#L326
		pass

	def save_to_file(self, filename, episode_idx, saver, sess):
		filename = filename + "." + str(self.model.method_type) + "." + str(episode_idx)
		saver.save(sess,filename)
