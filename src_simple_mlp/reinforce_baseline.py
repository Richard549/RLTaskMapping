import copy, random, math, os, logging
import util

import tensorflow as tf
from replay_memory import *
from agent import *
from policy_network import *
from state_value_network import *
from util import *

experiment_folder = sys.argv[1]
if "RES_DEBUG" in os.environ and os.environ["RES_DEBUG"] == "1":
	util.initialise_logging(experiment_folder + "/log.txt",True)
else:
	util.initialise_logging(experiment_folder + "/log.txt",False)

np.set_printoptions(suppress=True)

MODEL_CHECKPOINT_PERIOD = 1 # period given in units of episodes
MODEL_SAVE_FILENAME = "./models/model.h5"
DISCOUNT_FACTOR = 0.95 # i.e. no discount, we get the total reward to the end

VARIABLE_SUPPORT = False
ZERO_NON_INCLUDED_NODES = False
SPARSE_ADJ = False
INCLUDE_PARTIAL_SOLUTION_FEATURE = True

NUM_EPISODES = 200000
NUM_REPEATS = 1
NUM_STATS_EPISODES = 2
NUM_EPISODES_PER_UPDATE = 4

MINIBATCH_SIZE = 8
GCN_HIDDEN_NEURONS = 16 # hidden neurons during convolution
GCN_OUTPUT_NEURONS = 16 # how many neurons for a single node representation after convolving
GRAPH_POOLING_NEURONS = 64 # how many neurons for a whole graph representation

NUM_GCN_LAYERS = 1
NUM_DENSE_LAYERS = 3 # 2 dense layers and then one final

POLICY_LR = 0.0001
STATE_LR  = 0.0005

saved_execution_times_prefix = sys.argv[2] # will be appended by .NUM_REPEATS

execution_features = ["papi_tot_cyc"]

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('gcn_hidden', GCN_HIDDEN_NEURONS, 'Number of units in each hidden layer.')
flags.DEFINE_integer('gcn_output', GCN_OUTPUT_NEURONS, 'Number of units in each hidden layer.')
flags.DEFINE_integer('pooling_hidden', GRAPH_POOLING_NEURONS, 'Number of units in each hidden layer.')
flags.DEFINE_integer('num_simultaneous_graphs', MINIBATCH_SIZE, 'Number of graphs that can be input simultaneously')
flags.DEFINE_string('model', "SIMPLE_MLP", 'What model to use (GCN or MLP)')

flags.DEFINE_integer('num_gcn_layers', NUM_GCN_LAYERS, 'Number of layers in model')
flags.DEFINE_integer('num_dense_layers', NUM_DENSE_LAYERS, 'Number of layers in model')

benchmark = "/var/shared/openstream/examples/jacobi-2d/stream_df_jacobi_2d_GPU -s y14 -s x14 -b y11 -b x11 -r 1"
benchmark = "/var/shared/openstream/examples/jacobi-1d/stream_df_jacobi_1d_GPU -s x26 -b x22 -r 1"

adjacency_matrix_filename = "/var/shared/openstream/examples/jacobi-2d/14_11_1_dense_adj_matrix.csv"
adjacency_matrix_filename = "/var/shared/openstream/examples/jacobi-1d/26_22_1_adj.txt"

feature_matrix_filename = "/var/shared/openstream/examples/jacobi-2d/14_11_1_instances.csv"
feature_matrix_filename = "/var/shared/openstream/examples/jacobi-1d/26_22_1_instances.csv"

if experiment_folder.startswith("/"):
	output_schedule_filename = str(experiment_folder) + "/schedules/schedule.csv"
	MODEL_SAVE_FILENAME = experiment_folder + "/" + MODEL_SAVE_FILENAME
else: # relative
	p = subprocess.Popen("echo `pwd`/" + str(experiment_folder) + "/schedules/schedule.csv",stdout=subprocess.PIPE,shell=True)
	output, err = p.communicate()
	output_schedule_filename = output.decode("utf-8").replace("\n","")
	MODEL_SAVE_FILENAME = "./" + experiment_folder + "/" + MODEL_SAVE_FILENAME

util.ensure_dir_exists(output_schedule_filename)
util.ensure_dir_exists(MODEL_SAVE_FILENAME)

logging.debug("Initializing...")

logging.debug("Agent...")
agent = Agent(saved_execution_times_prefix, adjacency_matrix_filename, feature_matrix_filename, benchmark, execution_features, output_schedule_filename=output_schedule_filename, adjacency_is_sparse=SPARSE_ADJ, num_repeats=NUM_REPEATS)

logging.debug("Replay Memory...")
num_update_transitions = len(agent.all_nodes_to_allocate)*NUM_EPISODES_PER_UPDATE
replay_memory = ReplayMemory(num_update_transitions)

logging.debug("Policy Network...")
with tf.variable_scope('policy_network'):
	flags.DEFINE_string('method_type', "reinforce_policy", "'value' or 'policy' based policy_network")
	flags.DEFINE_float('learning_rate', POLICY_LR, 'Initial learning rate.')
	policy_network = ReinforcePolicyNetwork_SimpleMLP(agent.undirected_adj, agent.feature_matrix.shape[0], agent.feature_matrix.shape[1], agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
	policy_saver = tf.train.Saver([v for v in tf.all_variables() if 'policy_network' in v.name])
	
logging.debug("Value Network...")
tf.flags.FLAGS.__delattr__('method_type')
tf.flags.FLAGS.__delattr__('learning_rate')
flags.DEFINE_string('method_type', "batched_statevalue", "'value' or 'policy' or 'statevalue' network")
flags.DEFINE_float('learning_rate', STATE_LR, 'Initial learning rate.')
with tf.variable_scope('statevalue_network'):
	value_network = BatchedStateValueNetwork(agent.undirected_adj, agent.feature_matrix.shape[0], agent.feature_matrix.shape[1], agent.actions_vector)
	value_saver = tf.train.Saver([v for v in tf.all_variables() if 'statevalue_network' in v.name])

logging.debug("Tensorflow session...")
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth=True
sess = tf.Session(config=session_config)
sess.run(tf.global_variables_initializer())

logging.debug("Finished initialization")

tf.get_default_graph().finalize()

taking_stats = True
discounted_cumulative_rewards_stats = None

num_stats_transitions = len(agent.all_nodes_to_allocate)*NUM_STATS_EPISODES
print("Num stats transitions = " + str(num_stats_transitions))
replay_memory_stats = ReplayMemory(num_stats_transitions)

for episode_idx in range(NUM_EPISODES):

	termination = False
	step_idx = 0
	while termination == False:
		# so model input is all nodes of the state, model output is all (num_nodes, action_dim) logits
		# softmax is applied across all (num_nodes * action_dim) logits to give a probability distribution across (num_nodes * action_dim) that sums to 1
		actionable_nodes_at_current_state = np.copy(agent.get_actionable_nodes_at_state(agent.current_state))

		current_state_value = value_network.get_value_for_state(sess, agent.current_state.feature_matrix, actionable_nodes_at_current_state)

		action, probabilities = policy_network.get_best_action(sess, agent.current_state, actionable_nodes_at_current_state, agent.actions_vector)

		logging.debug("Taking " + str(action.label) + " after probabilities at step " + str(step_idx) + ":" + str(probabilities))
		step_idx += 1

		pre_state = copy.deepcopy(agent.current_state)
		new_state, reward, termination, _, _ = agent.take_action(action, episode_idx)
		
		if termination:
			logging.info("Episode reward: " + str(reward))

		replay_memory.add(Transition(pre_state, action, new_state, reward, termination, actionable_nodes_at_pre_state=actionable_nodes_at_current_state,action_probability=probabilities,pre_state_value=current_state_value))
		replay_memory_stats.add(Transition(pre_state, action, new_state, reward, termination, actionable_nodes_at_pre_state=actionable_nodes_at_current_state,pre_state_value=current_state_value))

	# go through the transitions of the replay memory in order
		# calculate the discounted reward of the transition
	episode_replay = replay_memory.get_ordered_transitions()

	discounted_cumulative_rewards = np.zeros(len(episode_replay))
	running_total_cumulative_reward = 0.0
	for i, transition in reversed(list(enumerate(episode_replay))):
		running_total_cumulative_reward = DISCOUNT_FACTOR * running_total_cumulative_reward + transition.reward
		discounted_cumulative_rewards[i] = running_total_cumulative_reward

	if taking_stats == True:
		stats_replay = replay_memory_stats.get_ordered_transitions()
		if len(stats_replay) > len(agent.all_nodes_to_allocate):
			discounted_cumulative_rewards_stats = np.zeros(len(stats_replay))
			running_total_cumulative_reward_stats = 0.0
			for i, transition in reversed(list(enumerate(stats_replay))):
				running_total_cumulative_reward_stats = DISCOUNT_FACTOR * running_total_cumulative_reward_stats + transition.reward
				discounted_cumulative_rewards_stats[i] = running_total_cumulative_reward_stats

			#discounted_cumulative_rewards = (discounted_cumulative_rewards - np.mean(discounted_cumulative_rewards_stats)) / np.std(discounted_cumulative_rewards_stats)
			print("Taking stats currently discounted cumulative rewards: " + str(discounted_cumulative_rewards))

		if len(stats_replay) >= num_stats_transitions:
			taking_stats = False

	else:
		#discounted_cumulative_rewards = (discounted_cumulative_rewards - np.mean(discounted_cumulative_rewards_stats)) / np.std(discounted_cumulative_rewards_stats)
		print("Discounted cumulative rewards: " + str(discounted_cumulative_rewards))

	indexes = np.arange(len(episode_replay))
	random.shuffle(indexes)

	if taking_stats == False and (episode_idx > 0) and (episode_idx % NUM_EPISODES_PER_UPDATE == 0):
		for start_index in range(0, len(episode_replay), MINIBATCH_SIZE):

			logging.info("New minibatch starting at " + str(start_index));
			minibatch_indexes = indexes[start_index:start_index + MINIBATCH_SIZE]

			if len(minibatch_indexes) != MINIBATCH_SIZE:
				logging.info("There is an unprocessed remainder with " + str(len(indexes)) + " transitions and a minibatch size of " + str(MINIBATCH_SIZE))
			else:
				features_per_transition = [transition.pre_state.feature_matrix for i, transition in enumerate(episode_replay) if i in minibatch_indexes]
				probability_distributions = [transition.action_probability for i, transition in enumerate(episode_replay) if i in minibatch_indexes]
				rewards = [reward for i, reward in enumerate(discounted_cumulative_rewards) if i in minibatch_indexes]
				baseline_state_values = [transition.pre_state_value for i, transition in enumerate(episode_replay) if i in minibatch_indexes]
				advantages = [reward - state_value for reward, state_value in list(zip(rewards,baseline_state_values))]

				actions = [transition.action for i, transition in enumerate(episode_replay) if i in minibatch_indexes]
				action_labels = [transition.action.label for i, transition in enumerate(episode_replay) if i in minibatch_indexes]
				action_nodes = [transition.action.node_idx for i, transition in enumerate(episode_replay) if i in minibatch_indexes]
				actionable_nodes_per_transition = [transition.actionable_nodes_at_pre_state for i, transition in enumerate(episode_replay) if i in minibatch_indexes]

				logging.info("For minibatch starting at " + str(start_index) + " the actions taken were nodes " + str(action_nodes) + " to labels " + str(action_labels));
				logging.info("For minibatch starting at " + str(start_index) + " the prior probabilities were: " + str(probability_distributions));
				logging.info("For minibatch starting at " + str(start_index) + " the rewards were: " + str(rewards));
				logging.info("For minibatch starting at " + str(start_index) + " the advantages were: " + str(advantages));

				loss = policy_network.fit_to_minibatch(sess, features_per_transition, actions, advantages, actionable_nodes_per_transition, None, None)
				logging.debug("Policy loss:" + str(loss))
					
				# train value network on minibatch
				value_loss = value_network.fit_to_minibatch(sess, features_per_transition, action_nodes, rewards)
				logging.debug("Value loss: " + str(value_loss))

				new_probability_distributions = [policy_network.get_best_action(sess, transition.pre_state, transition.actionable_nodes_at_pre_state, agent.actions_vector)[1] for i, transition in enumerate(episode_replay) if i in minibatch_indexes]

				logging.info("For minibatch starting at " + str(start_index) + " the post probabilities were: " + str(new_probability_distributions));

		if episode_idx > 0 and episode_idx % MODEL_CHECKPOINT_PERIOD == 0:
			logging.debug("Saving to file " + str(MODEL_SAVE_FILENAME))
			policy_network.save_to_file(MODEL_SAVE_FILENAME, episode_idx, policy_saver, sess)
			value_network.save_to_file(MODEL_SAVE_FILENAME, episode_idx, value_saver, sess)

	agent.reset()
	if (episode_idx > 0) and (episode_idx % NUM_EPISODES_PER_UPDATE == 0):
		replay_memory.reset()

sess.close()
logging.info("Complete")
