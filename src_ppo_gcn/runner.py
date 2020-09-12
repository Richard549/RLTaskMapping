import copy, random, math, os, logging, sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import util
import multiprocessing
import pickle

from multiprocessing import Pool

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from replay_memory import *
from agent import *
import state_value_network
import policy_network 
import rnd_network 
from util import *

from runner import *
	
MODEL_SAVE_FILENAME = "./models/model.h5"
NUM_EPISODES = 100000
TIMEOUT_SECS = 2
PARTIAL_REWARDS = 0 # how many evenly-spaced executions to take per episode (set to 0 to only take final execution)
DISCOUNT_FACTOR = 0.95 # we care much more about the final reward than we do about the partial rewards
LAMBDA = 0.95
LAMBDA_RETURNS = True
NORMALIZE_GAE = True
NORMALIZE_INTRINSIC = False
VARIABLE_SUPPORT = False # if theres no variation then I should include partial solution feature
ZERO_NON_INCLUDED_NODES = False
INCLUDE_PARTIAL_SOLUTION_FEATURE = True
ENABLE_RND = False
RND_COLLECTION_EPISODES = 10
SPARSE_ADJ = True

NUM_REPEATS = 1
NUM_POLICY_SAMPLES = 3
NUM_SIMULTANEOUS_POLICY_SAMPLES = 3
NUM_EPOCHS_PER_POLICY = 1
NUM_MINIBATCHES = 8
TARGET_POLICY_LR = 0.0003
TARGET_SV_LR = 0.0003
PPO_CLIP = 0.2

GCN_HIDDEN_NEURONS = 16 # hidden neurons during convolution
GCN_OUTPUT_NEURONS = 16 # how many neurons for a single node representation after convolving
GRAPH_POOLING_NEURONS = 64 # how many neurons for a whole graph representation
NUM_GCN_LAYERS = 2
NUM_DENSE_LAYERS = 2 # pooling and final

# so that each episode carries out an equivalent change to the parameters. regardless of how many samples we take
POLICY_LR = TARGET_POLICY_LR / (NUM_EPOCHS_PER_POLICY * NUM_MINIBATCHES)
SV_LR = TARGET_SV_LR / (NUM_EPOCHS_PER_POLICY * NUM_MINIBATCHES)

# Normalize advantages
NUM_EPISODES_COLLECT_STATS = 50 # after the first 50 episodes, we stop
sum_of_x = 0.0
sum_of_x2 = 0.0
num_transitions = 0

PATIENCE = 30
num_episodes_no_reward_change = 1
previous_different_execution_reward = 0.0

# this is ridiculously hacky but seems like the only way for the subprocesses to have their own loaded tensorflow models
subprocess_initialized = False
subprocess_value_network = None
subprocess_policy_network = None
subprocess_rnd_target_network = None
subprocess_rnd_predictor_network = None
subprocess_value_saver = None
subprocess_policy_saver = None
subprocess_rnd_target_saver = None
subprocess_rnd_predictor_saver = None
subprocess_session = None

def run_policy(experiment_folder, episode_idx, sample_idx, discount_factor, load_model_filename, minibatch_size, enable_rnd, lambda_returns, debugging, profiling_mode, policy_lr, batched_statevalue_lr, ppo_clip, gcn_hidden_neurons, gcn_output_neurons, pooling_hidden_neurons, num_gcn_layers, num_dense_layers, variable_scope, zero_non_included_nodes):

	# because we spawn rather than fork, everything has to be loaded from fresh...
	import copy, random, logging, time, traceback
	import util
	
	if subprocess_initialized == False:
		if debugging:
			util.initialise_logging(experiment_folder + "/log.txt",True)
		else:
			util.initialise_logging(experiment_folder + "/log.txt",False)
	logging.info("Running subprocess run_policy")

	import numpy as np
	np.set_printoptions(threshold=sys.maxsize)

	import tensorflow as tf
	from replay_memory import Transition
	import agent
	import state_value_network
	import policy_network 

	start = time.time()
	
	subprocess_agent = pickle.load(open("agent.pickle", "rb"))

	global subprocess_initialized
	global subprocess_value_network
	global subprocess_value_saver
	global subprocess_policy_network
	global subprocess_policy_saver
	global subprocess_rnd_target_network
	global subprocess_rnd_target_saver
	global subprocess_rnd_predictor_network
	global subprocess_rnd_predictor_saver
	global subprocess_session

	if subprocess_initialized == False:

		flags = tf.app.flags
		FLAGS = flags.FLAGS
		flags.DEFINE_float('e', ppo_clip, 'PPO clip.')
		flags.DEFINE_integer('gcn_hidden', gcn_hidden_neurons, 'Number of units in each hidden layer.')
		flags.DEFINE_integer('gcn_output', gcn_output_neurons, 'Number of units in each hidden layer.')
		flags.DEFINE_integer('pooling_hidden', pooling_hidden_neurons, 'Number of units in each hidden layer.')
		flags.DEFINE_integer('num_simultaneous_graphs', minibatch_size, 'Number of graphs that can be input simultaneously')
		flags.DEFINE_integer('num_gcn_layers', num_gcn_layers, 'Number of layers in model')
		flags.DEFINE_integer('num_dense_layers', num_dense_layers, 'Number of layers in model')
		flags.DEFINE_string('model', "GCN", 'What model to use (GCN or MLP)')

		session_config = tf.ConfigProto()
		session_config.gpu_options.allow_growth=True
		subprocess_session = tf.Session(config=session_config)

		logging.debug("Subprocess " + str(sample_idx) + " loading networks.")

		flags.DEFINE_string('method_type', "ppo_policy", "'value' or 'policy' based policy_network")
		flags.DEFINE_float('learning_rate', policy_lr, 'Initial learning rate.')
		with tf.variable_scope('policy_network'):
			subprocess_policy_network = policy_network.PPONetwork(subprocess_agent.undirected_adj, subprocess_agent.feature_matrix.shape[0], subprocess_agent.feature_matrix.shape[1], subprocess_agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
			subprocess_policy_saver = tf.train.Saver([v for v in tf.all_variables() if 'policy_network' in v.name])

		tf.flags.FLAGS.__delattr__('method_type')
		tf.flags.FLAGS.__delattr__('learning_rate')
		flags.DEFINE_string('method_type', "batched_statevalue", "'value' or 'policy' or 'statevalue' network")
		flags.DEFINE_float('learning_rate', batched_statevalue_lr, 'Initial learning rate.')
		with tf.variable_scope('statevalue_network'):
			subprocess_value_network = state_value_network.BatchedStateValueNetwork(subprocess_agent.undirected_adj, subprocess_agent.feature_matrix.shape[0], subprocess_agent.feature_matrix.shape[1], subprocess_agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
			subprocess_value_saver = tf.train.Saver([v for v in tf.all_variables() if 'statevalue_network' in v.name])
		
		if enable_rnd:
			tf.flags.FLAGS.__delattr__('method_type')
			flags.DEFINE_string('method_type', "rnd_target", "'value' or 'policy' or 'statevalue' network")
			with tf.variable_scope('rnd_target'):
				subprocess_rnd_target_network = state_value_network.BatchedStateValueNetwork(subprocess_agent.undirected_adj, subprocess_agent.feature_matrix.shape[0], subprocess_agent.feature_matrix.shape[1], subprocess_agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
				subprocess_rnd_target_saver = tf.train.Saver([v for v in tf.all_variables() if 'rnd_target' in v.name])
			
			tf.flags.FLAGS.__delattr__('method_type')
			flags.DEFINE_string('method_type', "rnd_predictor", "'value' or 'policy' or 'statevalue' network")
			with tf.variable_scope('rnd_predictor'):
				subprocess_rnd_predictor_network = state_value_network.BatchedStateValueNetwork(subprocess_agent.undirected_adj, subprocess_agent.feature_matrix.shape[0], subprocess_agent.feature_matrix.shape[1], subprocess_agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
				subprocess_rnd_predictor_saver = tf.train.Saver([v for v in tf.all_variables() if 'rnd_predictor' in v.name])
	
		subprocess_initialized = True

	# I need to load the previous state of the networks!
	subprocess_policy_network.load_from_file(load_model_filename, str(episode_idx-1), subprocess_policy_saver, subprocess_session)
	subprocess_value_network.load_from_file(load_model_filename, str(episode_idx-1), subprocess_value_saver, subprocess_session)
	if enable_rnd:
		subprocess_rnd_predictor_network.load_from_file(load_model_filename, str(episode_idx-1), subprocess_rnd_predictor_saver, subprocess_session)
		subprocess_rnd_target_network.load_from_file(load_model_filename, str(-1), subprocess_rnd_target_saver, subprocess_session) # TODO I only need to load this once!
	
	tf.get_default_graph().finalize()

	end = time.time()
	logging.info("Building everything in subprocess took " + str(end-start))
	logging.info("Running episode " + str(episode_idx) + " sample " + str(sample_idx))

	sampled_policy_transitions = []
	transitions_to_execute = []

	actionable_nodes_at_current_state = np.copy(subprocess_agent.get_actionable_nodes_at_state(subprocess_agent.base_state))
	actioned_nodes_at_current_state = np.copy(subprocess_agent.base_state.partial_solution_node_indexes)

	logging.info("Actioned nodes at base state:" + str(actioned_nodes_at_current_state))

	if profiling_mode: logging.info("BTime:state_value_predict_" + str(sample_idx))
	current_state_value = subprocess_value_network.get_value_for_state(subprocess_session, subprocess_agent.current_state.feature_matrix, actioned_nodes_at_current_state)
	#current_state_value = 0.0 # TODO debugging
	if profiling_mode: logging.info("ATime:state_value_predict_" + str(sample_idx))
	
	subprocess_agent.reset()

	termination = False
	step_idx = 0
	while termination == False:

		logging.info("Actionable nodes at step " + str(step_idx) + ":" + str(actionable_nodes_at_current_state))
		logging.info("Sample " + str(sample_idx) + " actionable node indexes at step " + str(step_idx) + ": " + ":".join(list(map(str,np.array(actionable_nodes_at_current_state)))))
		logging.info("Sample " + str(sample_idx) + " actionable node labels: " + ":".join(list(map(str,np.array(subprocess_agent.task_labels)[actionable_nodes_at_current_state]))))

		if profiling_mode: logging.info("BTime:policy_predict_" + str(sample_idx))
		current_action, action_probability = subprocess_policy_network.get_best_action(subprocess_session, subprocess_agent.current_state, actionable_nodes_at_current_state, subprocess_agent.actions_vector, sample_idx=sample_idx)
		if profiling_mode: logging.info("ATime:policy_predict_" + str(sample_idx))
		
		logging.debug("Sample " + str(sample_idx) + ":current state value is " + str(current_state_value))

		# take step according to policy
		pre_state = copy.deepcopy(subprocess_agent.current_state)
		new_state, reward, termination, should_execute, allocated_tasks = subprocess_agent.take_action(current_action, episode_idx, sample_idx=sample_idx, transition_idx=step_idx, delay_execution=True) # allocated tasks may be None if not executed
	
		if enable_rnd:
			partial_solution_indexes = np.copy(new_state.partial_solution_node_indexes)

			rnd_target_output = subprocess_rnd_target_network.get_value_for_state(subprocess_session, new_state.feature_matrix, partial_solution_indexes)
			rnd_prediction_output = subprocess_rnd_predictor_network.get_value_for_state(subprocess_session, new_state.feature_matrix, partial_solution_indexes)
			intrinsic_reward = rnd_target_output - rnd_prediction_output
		else:
			intrinsic_reward = 0.0
			rnd_target_output = None

		#if sample_idx == 0:
		actionable_nodes_at_new_state = np.copy(subprocess_agent.get_actionable_nodes_at_state(subprocess_agent.current_state))
		actioned_nodes_at_new_state = np.copy(new_state.partial_solution_node_indexes)
		if profiling_mode: logging.info("BTime:state_value_predict_" + str(sample_idx))
		next_state_value = subprocess_value_network.get_value_for_state(subprocess_session, subprocess_agent.current_state.feature_matrix, actioned_nodes_at_new_state)
		if profiling_mode: logging.info("ATime:state_value_predict_" + str(sample_idx))

		logging.info("Sample " + str(sample_idx) + " took action [" + str(current_action.node_idx) + "," + str(current_action.label) + "]")
		logging.info("Current state value: " + str(current_state_value) + " and next state value: " + str(next_state_value) + ", and direct reward was " + str(reward))

		if termination == False and lambda_returns == False:
				reward = reward + (DISCOUNT_FACTOR * next_state_value)

		if lambda_returns:
			advantage = reward # TD error computed later in this case (so this is not really advantage yet!)
		else:
			advantage = reward - current_state_value
			logging.debug("Sample " + str(sample_idx) + ":advantage of action [" + str(current_action.node_idx) + "," + str(current_action.label) + "]:" + str(advantage))

		# save state, action, reward, and action_probability as policy_sample_transitions
		transition = Transition(pre_state, current_action, new_state, advantage, termination, action_probability=action_probability, actionable_nodes_at_pre_state=actionable_nodes_at_current_state, allocated_tasks=allocated_tasks, rnd_target_output=rnd_target_output, pre_state_value=current_state_value, post_state_value=next_state_value, intrinsic_reward=intrinsic_reward)

		sampled_policy_transitions.append(transition)
		if should_execute:
			transitions_to_execute.append(step_idx)

		if termination == False:

			actionable_nodes_at_current_state = actionable_nodes_at_new_state
			current_state_value = next_state_value

		step_idx += 1

	return sampled_policy_transitions, transitions_to_execute

######################################################################################################################

if __name__ == "__main__":
	multiprocessing.set_start_method('spawn', force=True)
	multiprocessing.freeze_support()

	experiment_folder = sys.argv[1]
	if "RES_DEBUG" in os.environ and os.environ["RES_DEBUG"] == "1":
		debugging_mode = True
		util.initialise_logging(experiment_folder + "/log.txt",True)
	else:
		debugging_mode = False
		util.initialise_logging(experiment_folder + "/log.txt",False)
	
	profiling_mode = False
	if "RES_PROFILE" in os.environ and os.environ["RES_PROFILE"] == "1":
		profiling_mode = True

	saved_execution_times_prefix = sys.argv[2] # will be appended by .NUM_REPEATS

	execution_features = ["PAPI_TOT_INS","label"]

	flags = tf.app.flags
	FLAGS = flags.FLAGS

	benchmark = "/var/shared/openstream/examples/jacobi-1d/stream_df_jacobi_1d_GPU -s x26 -b x22 -r 1"

	logging.info("Exp:Running experiment with configuration:")
	logging.info("Exp:Benchmark:" + str(benchmark))
	logging.info("Exp:Discount factor:" + str(DISCOUNT_FACTOR))
	logging.info("Exp:Partial rewards:" + str(PARTIAL_REWARDS))
	logging.info("Exp:Num policy samples:" + str(NUM_POLICY_SAMPLES))
	logging.info("Exp:Num epochs per policy:" + str(NUM_EPOCHS_PER_POLICY))
	logging.info("Exp:Minibatches per epoch:" + str(NUM_MINIBATCHES))
	logging.info("Exp:Policy Target/Actual LR:" + str(TARGET_POLICY_LR) + "/" + str(POLICY_LR))
	logging.info("Exp:StateValue Target/Actual LR:" + str(TARGET_SV_LR) + "/" + str(SV_LR))
	logging.info("Exp:PPO clip:" + str(PPO_CLIP))
	logging.info("Exp:RND enabled:" + str(ENABLE_RND))
	logging.info("Exp:Lambda returns:" + str(LAMBDA_RETURNS))
	logging.info("Exp:Lambda:" + str(LAMBDA))

	adjacency_matrix_filename = "/var/shared/openstream/examples/jacobi-1d/outputs/26_22_1_dense_adjacency_matrix.csv"
	feature_matrix_filename = "/var/shared/openstream/examples/jacobi-1d/outputs/26_22_1_feature_matrix.csv"

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
	agent = Agent(saved_execution_times_prefix, adjacency_matrix_filename, feature_matrix_filename, benchmark, execution_features, output_schedule_filename=output_schedule_filename, adjacency_is_sparse=SPARSE_ADJ, num_repeats=NUM_REPEATS, timeout_secs=TIMEOUT_SECS, partial_rewards=PARTIAL_REWARDS)

	MINIBATCH_SIZE = math.ceil(float(agent.num_required_transitions*NUM_POLICY_SAMPLES)/ NUM_MINIBATCHES)
	logging.info("Exp:Minibatch size:" + str(MINIBATCH_SIZE))
	
	flags.DEFINE_float('e', PPO_CLIP, 'PPO clip.')
	flags.DEFINE_integer('gcn_hidden', GCN_HIDDEN_NEURONS, 'Number of units in each hidden layer.')
	flags.DEFINE_integer('gcn_output', GCN_OUTPUT_NEURONS, 'Number of units in each hidden layer.')
	flags.DEFINE_integer('pooling_hidden', GRAPH_POOLING_NEURONS, 'Number of units in each hidden layer.')
	flags.DEFINE_integer('num_simultaneous_graphs', MINIBATCH_SIZE, 'Number of graphs that can be input simultaneously')
	flags.DEFINE_integer('num_gcn_layers', NUM_GCN_LAYERS, 'Number of layers in model')
	flags.DEFINE_integer('num_dense_layers', NUM_DENSE_LAYERS, 'Number of layers in model')
	flags.DEFINE_string('model', "GCN", 'What model to use (GCN or MLP)')

	logging.debug("Policy Network...")
	flags.DEFINE_string('method_type', "ppo_policy", "'value' or 'policy' based policy_network")
	flags.DEFINE_float('learning_rate', POLICY_LR, 'Initial learning rate.')
	with tf.variable_scope('policy_network'):
		policy_network = policy_network.PPONetwork(agent.undirected_adj, agent.feature_matrix.shape[0], agent.feature_matrix.shape[1], agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
		policy_saver = tf.train.Saver([v for v in tf.all_variables() if 'policy_network' in v.name])

	logging.debug("Value Network...")
	tf.flags.FLAGS.__delattr__('method_type')
	tf.flags.FLAGS.__delattr__('learning_rate')
	flags.DEFINE_string('method_type', "batched_statevalue", "'value' or 'policy' or 'statevalue' network")
	flags.DEFINE_float('learning_rate', SV_LR, 'Initial learning rate.')
	with tf.variable_scope('statevalue_network'):
		value_network = state_value_network.BatchedStateValueNetwork(agent.undirected_adj, agent.feature_matrix.shape[0], agent.feature_matrix.shape[1], agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
		value_saver = tf.train.Saver([v for v in tf.all_variables() if 'statevalue_network' in v.name])

	if ENABLE_RND:
		tf.flags.FLAGS.__delattr__('method_type')
		flags.DEFINE_string('method_type', "rnd_target", "'value' or 'policy' or 'statevalue' network")
		with tf.variable_scope('rnd_target'):
			rnd_target_network = state_value_network.BatchedStateValueNetwork(agent.undirected_adj, agent.feature_matrix.shape[0], agent.feature_matrix.shape[1], agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
			rnd_target_saver = tf.train.Saver([v for v in tf.all_variables() if 'rnd_target' in v.name])
		
		tf.flags.FLAGS.__delattr__('method_type')
		flags.DEFINE_string('method_type', "rnd_predictor", "'value' or 'policy' or 'statevalue' network")
		with tf.variable_scope('rnd_predictor'):
			rnd_predictor_network = state_value_network.BatchedStateValueNetwork(agent.undirected_adj, agent.feature_matrix.shape[0], agent.feature_matrix.shape[1], agent.actions_vector, include_partial_solution_feature=INCLUDE_PARTIAL_SOLUTION_FEATURE, zero_non_included_nodes=ZERO_NON_INCLUDED_NODES, variable_support=VARIABLE_SUPPORT)
			rnd_predictor_saver = tf.train.Saver([v for v in tf.all_variables() if 'rnd_predictor' in v.name])

	logging.debug("Tensorflow session...")
	session_config = tf.ConfigProto()
	session_config.gpu_options.allow_growth=True

	sess = tf.Session(config=session_config)

	sess.run(tf.global_variables_initializer())
	
	logging.debug("Finished initialization")
	tf.get_default_graph().finalize()

	# save the base agent for the subprocesses to use (rather than loading a new agent in each subprocess)
	pickle.dump(agent, open("agent.pickle", "wb"))

	logging.debug("Saving base networks:")
	value_network.save_to_file(MODEL_SAVE_FILENAME, "-1", value_saver, sess)
	policy_network.save_to_file(MODEL_SAVE_FILENAME, "-1", policy_saver, sess)
	if ENABLE_RND:
		rnd_target_network.save_to_file(MODEL_SAVE_FILENAME, "-1", rnd_target_saver, sess)
		rnd_predictor_network.save_to_file(MODEL_SAVE_FILENAME, "-1", rnd_predictor_saver, sess)

	pool = Pool(processes=NUM_SIMULTANEOUS_POLICY_SAMPLES)

	mean_advantage = 0.0
	stdev_advantage = 0.0

	stop = False
	for episode_idx in range(NUM_EPISODES):

		if stop == True:
			break

		# episode_transitions
		policy_transitions_per_sample = []
		transitions_to_execute_per_sample = [] # the agent returns transitions for which the post_state should be executed (the schedule file is ready to go, identified by episode and sample ids)

		if profiling_mode: logging.info("BTime:running_policies")
		# run the policy trajectories
		trajectories = []
		for policy_sample_idx in range(NUM_POLICY_SAMPLES):
			args = [experiment_folder, episode_idx, policy_sample_idx, DISCOUNT_FACTOR, MODEL_SAVE_FILENAME, MINIBATCH_SIZE, ENABLE_RND, LAMBDA_RETURNS, debugging_mode, profiling_mode, POLICY_LR, SV_LR, PPO_CLIP, GCN_HIDDEN_NEURONS, GCN_OUTPUT_NEURONS, GRAPH_POOLING_NEURONS, NUM_GCN_LAYERS, NUM_DENSE_LAYERS, VARIABLE_SUPPORT, ZERO_NON_INCLUDED_NODES]
			trajectory = pool.apply_async(run_policy, args)
	
			# If not multiprocessing then uncomment:
			#trajectory = run_policy_non_multiprocessing(experiment_folder, episode_idx, policy_sample_idx, DISCOUNT_FACTOR, MODEL_SAVE_FILENAME, MINIBATCH_SIZE, ENABLE_RND, LAMBDA_RETURNS, agent, value_network, policy_network, sess)
			trajectories.append(trajectory)

		# wait for them to complete and get their results
		for policy_sample_idx in range(NUM_POLICY_SAMPLES):
			# If not multiprocessing then swap the comment here:
			sampled_transitions, sample_transitions_to_execute = trajectories[policy_sample_idx].get()
			#sampled_transitions, sample_transitions_to_execute = trajectories[policy_sample_idx]
			policy_transitions_per_sample.append(sampled_transitions)
			transitions_to_execute_per_sample.append(sample_transitions_to_execute)

		if profiling_mode: logging.info("ATime:running_policies")

		# execute the trajectories
		if profiling_mode: logging.info("BTime:execute_and_process_all_samples")
		for policy_sample_idx in range(NUM_POLICY_SAMPLES):
			logging.info("Executing the " + str(len(transitions_to_execute_per_sample[policy_sample_idx])) + " necessary transitions for episode " + str(episode_idx) + " sample " + str(policy_sample_idx))

			if profiling_mode: logging.info("BTime:execute_and_process_sample")
			for solution_idx, transition_to_execute_idx in enumerate(transitions_to_execute_per_sample[policy_sample_idx]):
				
				schedule_filename = agent.output_schedule_filename + "." + str(episode_idx) + "." + str(policy_sample_idx) + "." + str(transition_to_execute_idx)
				allocated_tasks = policy_transitions_per_sample[policy_sample_idx][transition_to_execute_idx].allocated_tasks

				final_execution = False
				if solution_idx+1 == len(transitions_to_execute_per_sample[policy_sample_idx]):
					final_execution = True
				execution_reward = agent.execute_benchmark(schedule_filename, allocated_tasks, episode_idx, policy_sample_idx, transition_to_execute_idx, final_execution)

				# adjust the partial solution reward to reflect how partial it is...
				# normalize the partial rewards to be a function of their partial-ness (i.e. if a reward is -0.5 at a 20% partial solution, then make it -0.1 (i.e. reward * 20/100)
				if final_execution == False:
					execution_reward = (1.0/float(PARTIAL_REWARDS)) * execution_reward
				
				if execution_reward == previous_different_execution_reward:
					num_episodes_no_reward_change += 1
					if num_episodes_no_reward_change == PATIENCE:
						stop = True
						break
				else:
					num_episodes_no_reward_change = 1
					previous_different_execution_reward = execution_reward

				policy_transitions_per_sample[policy_sample_idx][transition_to_execute_idx].reward += execution_reward
			
			if stop == True:
				break

			if LAMBDA_RETURNS:
				# compute the lambda returns through the transitions of the sample

				if profiling_mode: logging.info("BTime:compute_lambda_returns")
				for i, transition in enumerate(policy_transitions_per_sample[policy_sample_idx]):

					returns = []

					current_cumulative_reward = transition.reward

					T = len(policy_transitions_per_sample[policy_sample_idx])
					final_n = 0
					for n in range(1,T-i):
						# n is how many forward rewards to go towards

						transition_n = policy_transitions_per_sample[policy_sample_idx][i+n]

						state_value_at_n = transition_n.pre_state_value

						n_step_return = current_cumulative_reward + (DISCOUNT_FACTOR**(n)) * state_value_at_n

						current_cumulative_reward += (DISCOUNT_FACTOR**(n)) * transition_n.reward

						n_step_return = LAMBDA**(n-1) * n_step_return

						returns.append(n_step_return)

						final_n = n 

					monte_carlo_reward = current_cumulative_reward

					lambda_return = (1.0-LAMBDA) * np.sum(returns) + LAMBDA**(final_n) * monte_carlo_reward
					logging.debug("Taking action [" + str(transition.action.node_idx) + "," + str(transition.action.label) + "] gave Lambda-Return:" + str(lambda_return))
					transition.reward = lambda_return
					transition.advantage = lambda_return - transition.pre_state_value
					logging.debug("Taking action [" + str(transition.action.node_idx) + "," + str(transition.action.label) + "] gave GAE:" + str(transition.advantage))

					if NORMALIZE_GAE:

						if episode_idx < NUM_EPISODES_COLLECT_STATS:
							num_transitions += 1						
							sum_of_x += transition.advantage
							sum_of_x2 += transition.advantage**2
							mean_advantage = sum_of_x / num_transitions
							stdev_advantage = np.sqrt((sum_of_x2 / num_transitions) - (mean_advantage * mean_advantage)) 

						logging.info("Mean transition advantage:" + str(mean_advantage))
						logging.info("Standard deviation transition advantage:" + str(stdev_advantage))

						if num_transitions > 1:
							transition.advantage = ((transition.advantage - mean_advantage) / stdev_advantage)

						logging.info("Taking action [" + str(transition.action.node_idx) + "," + str(transition.action.label) + "] gave normalized GAE:" + str(transition.advantage))

					action_allocation_str = "GPU" if transition.action.label == 1 else "CPU"
					logging.info("Sample " + str(policy_sample_idx) + " action " + str(i) + " as [" + str(transition.action.node_idx) + "," + str(transition.action.label) + "] (task " + str(agent.task_labels[transition.action.node_idx]) + " on " + action_allocation_str + ") with probability " + "{0:.3f}".format(transition.action_probability) + " gave GAE-advantage:" + str(transition.advantage))

			if profiling_mode: logging.info("ATime:compute_lambda_returns")
			if profiling_mode: logging.info("ATime:execute_and_process_sample")
		if profiling_mode: logging.info("ATime:execute_and_process_all_samples")

		all_policy_transitions = [transition for sample in policy_transitions_per_sample for transition in sample]
		
		agent.reset() # resetting in case I am debugging and using this agent (instead of multiprocessing)

		# Now that I have collected lots of policy samples... run minibatch gradient descent on them
		overall_policy_loss = 0.0
		overall_value_loss = 0.0
		overall_rnd_loss = 0.0
		if profiling_mode: logging.info("BTime:running_all_epochs")
		for epoch_idx in range(NUM_EPOCHS_PER_POLICY):

			logging.info("Running epoch " + str(epoch_idx) + " for episode " + str(episode_idx))

			random.shuffle(all_policy_transitions)

			if profiling_mode: logging.info("BTime:epoch")
			for start_index in range(0, len(all_policy_transitions), MINIBATCH_SIZE):

				minibatch = all_policy_transitions[start_index:start_index + MINIBATCH_SIZE]
				if len(minibatch) < MINIBATCH_SIZE:
					remaining_number = MINIBATCH_SIZE - len(minibatch)
					appended_transitions = np.random.choice(all_policy_transitions, remaining_number)
					minibatch.extend(appended_transitions) 

				features_per_transition = [transition.pre_state.feature_matrix for transition in minibatch]
				post_state_features_per_transition = [transition.post_state.feature_matrix for transition in minibatch]
				values = [transition.reward for transition in minibatch] # expected TD cumulative reward from the pre_state
				advantages = [transition.advantage for transition in minibatch] # advantage of taking this action in the pre_state
				actions = [transition.action for transition in minibatch]
				actionable_nodes_per_transition = [transition.actionable_nodes_at_pre_state for transition in minibatch]
				old_probabilities = [transition.action_probability for transition in minibatch]
				pre_actioned_nodes_per_transition = [transition.pre_state.partial_solution_node_indexes for transition in minibatch]
				
				logging.debug("The trajectory was: " + str([[transition.action.node_idx,transition.action.label] for transition in minibatch]))

				# train value network on minibatch
				if profiling_mode: logging.info("BTime:sv_fit")
				value_loss = 0.0
				if episode_idx >= RND_COLLECTION_EPISODES or ENABLE_RND == False: # if I want to collect statistics before fitting (but the statistics are relative to the model so they aren't constant anyway...)
					value_loss = value_network.fit_to_minibatch(sess, features_per_transition, pre_actioned_nodes_per_transition, values)
				if profiling_mode: logging.info("ATime:sv_fit")
				logging.debug("Value loss: " + str(value_loss))
				overall_value_loss += value_loss

				# train policy on minibatch
				if profiling_mode: logging.info("BTime:policy_fit")
				policy_loss = 0.0
				if episode_idx >= RND_COLLECTION_EPISODES or ENABLE_RND == False: # if I want to collect statistics before fitting (but the statistics are relative to the model so they aren't constant anyway...)
					policy_loss = policy_network.fit_to_minibatch(sess, features_per_transition, actions, advantages, actionable_nodes_per_transition, old_probabilities, pre_actioned_nodes_per_transition)
				if profiling_mode: logging.info("ATime:policy_fit")
				logging.debug("Policy loss: " + str(policy_loss))
				overall_policy_loss += policy_loss
				
				if ENABLE_RND and episode_idx >= RND_COLLECTION_EPISODES:
					# train the predictor network to minimize MSE with the target values from the minibatch transitions
					target_network_values = [transition.rnd_target_output for transition in minibatch]
					post_actioned_nodes_per_transition = [transition.post_state.partial_solution_node_indexes for transition in minibatch]

					logging.debug("The target network values are: " + str(target_network_values))
					logging.debug("The actioned nodes per transition are: " + str(post_actioned_nodes_per_transition))
					rnd_loss = rnd_predictor_network.fit_to_minibatch(sess, post_state_features_per_transition, post_actioned_nodes_per_transition, target_network_values)
					logging.info("RND loss: " + str(rnd_loss))
					overall_rnd_loss += rnd_loss

			if profiling_mode: logging.info("ATime:epoch")

		if profiling_mode: logging.info("ATime:running_all_epochs")
		logging.info("Overall policy loss: " + str(overall_policy_loss))
		logging.info("Overall value loss: " + str(overall_value_loss))
		if ENABLE_RND:
			logging.info("Overall RND loss: " + str(overall_rnd_loss))

		logging.debug("Saving updated networks:")
		if profiling_mode: logging.info("BTime:saving_networks")
		value_network.save_to_file(MODEL_SAVE_FILENAME, episode_idx, value_saver, sess)
		policy_network.save_to_file(MODEL_SAVE_FILENAME, episode_idx, policy_saver, sess)
		if ENABLE_RND:
			rnd_predictor_network.save_to_file(MODEL_SAVE_FILENAME, episode_idx, rnd_predictor_saver, sess)
		if profiling_mode: logging.info("ATime:saving_networks")
		
	sess.close()
	logging.info("Complete")
