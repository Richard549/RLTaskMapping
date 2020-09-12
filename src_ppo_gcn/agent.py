import copy
import numpy as np
import scipy.sparse as sp
import subprocess
import logging
import os
from subprocess import Popen, PIPE, TimeoutExpired
from threading import Timer
import signal

from gcn.utils import *

class Action():

	def __init__(self, node_idx=None, label=None):
		self.node_idx = node_idx
		self.label = label

class State():

	def __init__(self, feature_matrix, partial_solution_node_indexes=None, unselected_independent_node_indexes=None):

		self.feature_matrix = feature_matrix # always dense

		if partial_solution_node_indexes is None:
			self.partial_solution_node_indexes = []
		else:
			self.partial_solution_node_indexes = partial_solution_node_indexes

		if unselected_independent_node_indexes is None:
			self.unselected_independent_node_indexes = []
		else:
			self.unselected_independent_node_indexes = unselected_independent_node_indexes

	def get_schedule(self, final=True):

		nodes_in_solution = self.feature_matrix.transpose()[-1] # binary array, 1 meaning present in partial solution
		if final and 0 in nodes_in_solution:
			logging.debug("nodes_in_solution is: " + str(nodes_in_solution))
			logging.fatal("Asked to return schedule, yet not all tasks have been allocated.")
			exit(1)

		schedule = self.feature_matrix.transpose()[-2]
		return schedule
	
	def get_partial_schedule(self):

		node_allocations = self.feature_matrix.transpose()[-2]
		nodes_in_solution = self.feature_matrix.transpose()[-1] # binary array, 1 meaning present in partial solution

		for node_idx in range(len(nodes_in_solution)):
			if nodes_in_solution[node_idx] == 0:
				node_allocations[node_idx] = -1

		return node_allocations

class Agent():

	# directed_adj: adjacency matrix (N by N) where rows are consumers, and columns are producers. Therefore, a 1 in a row identifies a dependency of the row_idx on the column_idx
	# feature_matrix: node-feature matrix (N by M)
	
	def __init__(self,
			saved_execution_times_prefix,
			adjacency_matrix_filename,
			feature_matrix_filename,
			benchmark,
			execution_features,
			output_schedule_filename,
			adjacency_is_sparse=False,
			num_repeats=1,
			target_total_execution_count=1, # this is how many times a particular schedule will be executed, before we just use historical executions
			timeout_secs=10,
			partial_rewards=0 # how to divide the MDP into partial solutions i.e. 4 executes every 25%, 0 is ignore
			):

		self.output_schedule_filename = output_schedule_filename
		self.benchmark = benchmark
		self.num_repeats = num_repeats
		self.execution_times_filename = saved_execution_times_prefix+ "." + str(self.num_repeats)
		self.target_total_execution_count = target_total_execution_count
		self.timeout_secs = timeout_secs
		self.partial_rewards = partial_rewards

		self.mean_cpu_execution_time = None
		self.mean_gpu_execution_time = None
		self.load_saved_execution_times()

		logging.debug("Loading adjacency matrix in " + str(adjacency_matrix_filename))
		with open(adjacency_matrix_filename, 'r') as f:

			lines = [line.replace("\n","") for line in f]

			self.num_tasks = int(lines[0])
			self.task_labels = lines[1:self.num_tasks+1]

			self.directed_adj = np.array([list(map(int,row.split(","))) for row in lines[self.num_tasks+1:]],dtype=np.int16)
			self.adj_is_sparse = adjacency_is_sparse

			if self.adj_is_sparse:
				#transpose_directed_adj = self.directed_adj.transpose()
				#logging.info("transpose_directed_adj:" + str(transpose_directed_adj))
				#row = transpose_directed_adj[0]
				#col = transpose_directed_adj[1]
				#data = transpose_directed_adj[2]
				self.sparse_directed_adj = sp.coo_matrix(self.directed_adj)

				#self.independent_task_indexes = [node_index for node_index in range(self.num_tasks) if self.directed_adj.getrow(node_index).max() == 0]
				self.independent_task_indexes = np.setdiff1d(range(self.num_tasks), self.sparse_directed_adj.row)

			if True: # TODO debugging
				self.independent_task_indexes = [node_index for node_index in range(self.num_tasks) if np.amax(self.directed_adj[node_index]) == 0]

				self.undirected_adj = np.copy(self.directed_adj)
				for row in range(self.undirected_adj.shape[0]):
					for col in range(self.undirected_adj.shape[1]):
						edge = max(self.undirected_adj[row][col], self.undirected_adj[col][row])
						self.undirected_adj[row][col] = edge
						self.undirected_adj[col][row] = edge
		
		logging.info("There are " + str(len(self.independent_task_indexes)) + " tasks with no input dependences.")

		logging.debug("Loading feature matrix in " + str(feature_matrix_filename))
		feature_matrix = []
		non_eligible_nodes = []

		include_label = False
		if "label" in execution_features:
			include_label = True
			execution_features = [feature for feature in execution_features if feature != "label"]

		with open(feature_matrix_filename, 'r') as f:

			feature_matrix = [None] * self.num_tasks
			feature_indexes = []
			eligible_index = None

			line_idx = 0
			for line in f:
				line = line.replace("\n","")

				if line_idx == 0:
					header = line.split(",")
					feature_indexes = [index for index in range(len(header)) if header[index] in execution_features]
					eligible_index = [index for index in range(len(header)) if header[index].lower() == "GPU_ELIGIBLE".lower()]

					if len(eligible_index) != 1:
						logging.error("Feature matrix header does not contain the GPU eligibile flag: 'GPU_ELIGIBLE'")
						logging.error("Header = '" + str(header) + "'")
						exit(0)

					eligible_index = eligible_index[0]

					if len(feature_indexes) != len(execution_features):
						logging.error("Feature matrix header does not contain the requested execution features")
						logging.error("Header = '" + str(header) + "'")
						logging.fatal("Requested features = " + str(execution_features))
						exit(0)

					self.num_features = len(execution_features) + 2 # (+ 1 for binary device feature, + 1 for presence-in-partial-solution)

					line_idx += 1
					continue

				line = line.split(",")
				logging.debug(feature_indexes)
				logging.debug(line)

				if include_label:
					feature_matrix[line_idx-1] = [line_idx-1] + [int(line[index]) for index in feature_indexes] + [0, 0]
				else:
					feature_matrix[line_idx-1] = [int(line[index]) for index in feature_indexes] + [0, 0]

				if int(line[eligible_index]) == 0:
					# the task is not eligible for gpu execution
					non_eligible_nodes.append(line_idx-1)

				line_idx += 1
		
		self.feature_matrix = np.array(feature_matrix,dtype=np.float32)
		num_execution_features = len(execution_features)
		if include_label:
			num_execution_features += 1
		for i in range(num_execution_features):
			pass
			self.feature_matrix[:,i] = (self.feature_matrix[:,i] - np.mean(self.feature_matrix.transpose()[i])) / np.std(self.feature_matrix.transpose()[i])

		self.actions_vector = [0,1] # 0 = CPU, 1 = GPU

		non_eligible_nodes = np.array(non_eligible_nodes,dtype=np.int32)
		partial_solution_nodes  = non_eligible_nodes

		'''
		# now carry out some pre-allocations for debugging purposes:
		pre_allocated_nodes = []

		gpu_node_ranges = [[15,24],[29,38],[43,52]]
		gpu_nodes = []
		for arange in gpu_node_ranges:
			gpu_nodes += list(np.arange(arange[0],arange[1]))
		gpu_nodes = np.unique(np.array(gpu_nodes))

		constrained_action_space_ranges = [[38,43],[52,58]]
		constrained_action_space = []
		for arange in constrained_action_space_ranges:
			constrained_action_space += list(np.arange(arange[0],arange[1]))
		constrained_action_space = np.unique(np.array(constrained_action_space))

		eligible_nodes = np.setdiff1d(np.arange(self.num_tasks),non_eligible_nodes)
		constrained_action_space = np.setdiff1d(constrained_action_space, non_eligible_nodes)

		# the nodes to allocate to GPU are all the nodes that aren't in our constrained action space
		pre_allocated_cpu_nodes = np.setdiff1d(np.arange(self.num_tasks), constrained_action_space)
		self.feature_matrix[pre_allocated_cpu_nodes,-2] = 0
		self.feature_matrix[pre_allocated_cpu_nodes,-1] = 1

		pre_allocated_gpu_nodes = np.array([11,12,24,25,26])
		self.feature_matrix[pre_allocated_gpu_nodes,-2] = 1
		self.feature_matrix[pre_allocated_gpu_nodes,-1] = 1

		pre_allocated_nodes = np.concatenate((pre_allocated_cpu_nodes, pre_allocated_gpu_nodes))
		partial_solution_nodes = np.unique(np.concatenate((non_eligible_nodes, pre_allocated_nodes)))
		'''
		## END of debugging

		self.num_non_eligible_nodes = len(non_eligible_nodes)

		logging.info("There are " + str(len(non_eligible_nodes)) + " non_eligible_nodes.")
		logging.info("There are " + str(self.num_tasks) + " nodes in total.")
		self.num_required_transitions = self.num_tasks - len(non_eligible_nodes)
		logging.info("The objective is therefore to schedule for " + str(self.num_required_transitions) + " tasks.")

		# pre-select all non-eligible nodes as CPU tasks
		self.feature_matrix[non_eligible_nodes,-2] = 0
		self.feature_matrix[non_eligible_nodes,-1] = 1
		
		self.independent_task_indexes = np.setdiff1d(self.independent_task_indexes,non_eligible_nodes)

		self.base_state = State(np.copy(self.feature_matrix),unselected_independent_node_indexes=np.copy(self.independent_task_indexes),partial_solution_node_indexes=np.copy(partial_solution_nodes).tolist())
		self.current_state = State(np.copy(self.base_state.feature_matrix),unselected_independent_node_indexes=np.copy(self.base_state.unselected_independent_node_indexes),partial_solution_node_indexes=np.copy(self.base_state.partial_solution_node_indexes).tolist())
	
		self.all_nodes_to_allocate = np.setdiff1d(np.arange(self.num_tasks),non_eligible_nodes)

		num_eligible_nodes = len(self.get_actionable_nodes_at_state(self.current_state))
		logging.info("Num actionable GPU nodes at base state: " + str(num_eligible_nodes))

		if self.mean_cpu_execution_time is None or self.mean_gpu_execution_time is None:
			self.calculate_execution_stats()
		else:
			self.execution_time_variation = abs(self.mean_gpu_execution_time - self.mean_cpu_execution_time)

		self.mean_center = min([self.mean_cpu_execution_time, self.mean_gpu_execution_time]) + (self.execution_time_variation / 2.0) # center all rewards on the mid point between all_cpu and all_gpu

	def load_saved_execution_times(self):

		self.saved_execution_times = dict()
		if os.path.isfile(self.execution_times_filename):
			with open(self.execution_times_filename, 'r') as f:
				line_idx = 0
				for line in f:
					if line_idx == 0:
						benchmark_identifier = line.replace("\n","")
						if benchmark_identifier != self.benchmark:
							logging.fatal("The execution times recorded in " + str(self.execution_times_filename) + " are not for the current benchmark being run, which is: '" + str(self.benchmark) + "'")
							exit(1)
						line_idx += 1
					else:
						schedule_identifier = ":".join(line.split(":")[:-1])
						time = float(line.replace("\n","").split(":")[-1])
						if "mean_cpu" in schedule_identifier:
							self.mean_cpu_execution_time = time
							continue
						elif "mean_gpu" in schedule_identifier:
							self.mean_gpu_execution_time = time
							continue

						if schedule_identifier in self.saved_execution_times:
							self.saved_execution_times[schedule_identifier].append(time)
						else:
							self.saved_execution_times[schedule_identifier] = [time]

			for schedule_identifier in list(self.saved_execution_times.keys()):
				self.saved_execution_times[schedule_identifier] = np.array(self.saved_execution_times[schedule_identifier],dtype=np.float32)
				# dont convert
				#logging.debug("For schedule '" + str(schedule_identifier) + "' there are " + str(len(self.saved_execution_times[schedule_identifier])) + " saved execution times.")

		else:
			# write the first line as equal to the benchmark
			with open(self.execution_times_filename, 'w') as f:
				f.write(self.benchmark + "\n")

	def get_current_state(self):
		return self.current_state

	def get_actionable_nodes_at_state(self, state):

		already_selected_indexes = state.partial_solution_node_indexes

		# Can only action nodes that have their dependencies fulfilled
		'''		
		potential_nodes = self.sparse_directed_adj.row
		dependencies = self.sparse_directed_adj.col

		non_eligible_nodes = np.concatenate((potential_nodes[~np.in1d(dependencies,already_selected_indexes)],state.partial_solution_node_indexes))

		actionable_nodes = np.setdiff1d(potential_nodes, non_eligible_nodes)

		actionable_nodes = np.concatenate((actionable_nodes, state.unselected_independent_node_indexes))
		'''

		# all non-actioned nodes are actionable at all times
		'''
		actionable_nodes = np.setdiff1d(range(self.num_tasks), already_selected_indexes)
		'''

		# actionable nodes is the single next node in the list, so make a decision one at a time
		num_nodes_allocated_so_far = len(already_selected_indexes) - self.num_non_eligible_nodes

		if num_nodes_allocated_so_far == self.num_required_transitions:
			actionable_nodes = np.array([])
		else:
			actionable_nodes = np.array([self.all_nodes_to_allocate[num_nodes_allocated_so_far]])

		return actionable_nodes # should just be a list consisting of a single integer

	def get_next_random_action(self, actionable_nodes):
		actioned_node_index = np.random.choice(actionable_nodes, 1)[0]
		random_action = np.random.choice(self.actions_vector, 1)[0]
		return Action(actioned_node_index, random_action)

	def calculate_execution_stats(self):

		stats_schedule_file = self.output_schedule_filename + ".cpustats"

		with open(stats_schedule_file, 'w') as f:
			pass

		cmd = "OPENSTREAM_SCHEDULE_FILE=" + str(stats_schedule_file) + " " + str(self.benchmark)

		overall_execution_time = 0.0
		for repeat in range(self.num_repeats):
			failure = True
			while failure:
				try:
					logging.debug("Running repeat " + str(repeat) + " cmd '" + str(cmd) + "'")
					p = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True,universal_newlines=True, preexec_fn=os.setsid)
					try:
						stdout, stderr = p.communicate(timeout=self.timeout_secs) # will raise exception if fail
						logging.debug("Benchmark output: " + str(stdout) + "... and stderr:" + str(stderr))
						logging.debug("Returncode: " + str(p.returncode))
						if stderr is None and p.returncode == 0:
							failure = False
						else:
							logging.error("Benchmark did not execute successfully for cpu stats collection.")
							os.killpg(os.getpgid(p.pid), signal.SIGKILL)
					except TimeoutExpired:
						logging.error("Timeout expired during execution of schedule for cpu stats collection.")
						os.killpg(os.getpgid(p.pid), signal.SIGKILL)
				except ProcessLookupError:
					logging.error("ProcessLookupError caught. Continuing.")

			line_results = []
			running_line = ""
			for char in stdout:
				if char == '\n':
					line_results.append(running_line)
					running_line = ""
				else:
					running_line = running_line + char

			for line in line_results:
				try:
					execution_time = float(line)
					break
				except ValueError:
					continue
			
			overall_execution_time += execution_time

		execution_time = overall_execution_time
		self.mean_cpu_execution_time = execution_time # TODO not properly collecting mean yet

		with open(self.execution_times_filename, 'a') as f:
			f.write("mean_cpu:" + str(execution_time) + "\n")

		stats_schedule_file = self.output_schedule_filename + ".gpustats"

		schedule = self.base_state.feature_matrix.transpose()[-2]
		already_selected_indexes = np.array(self.base_state.partial_solution_node_indexes)
		actionable_nodes = np.array(list(range(len(schedule))))
		actionable_nodes = np.setdiff1d(actionable_nodes, already_selected_indexes)
		#actionable_nodes = np.concatenate((actionable_nodes, self.base_state.unselected_independent_node_indexes))

		allocated_tasks = []
		for i in range(len(schedule)):
			if i not in already_selected_indexes:
				allocated_tasks.append(i)
		
		with open(stats_schedule_file, 'w') as f:
			for task_idx in range(len(schedule)):
				if task_idx in allocated_tasks:
					#allocated_tasks.append(task_idx)
					gpu_device_allocation = 0
					f.write(str(self.task_labels[task_idx]) + ":" + str(gpu_device_allocation) + "\n")

		#scheduler_identifier = ":".join([self.task_labels[gpu_task_idx].replace("\n","") for gpu_task_idx in allocated_tasks])
		schedule_identifier = "mean_gpu"

		cmd = "OPENSTREAM_SCHEDULE_FILE=" + str(stats_schedule_file) + " " + str(self.benchmark)

		overall_execution_time = 0.0
		execution_time = 0.0
		for repeat in range(self.num_repeats):
			failure = True
			while failure:
				try:
					logging.debug("Running repeat " + str(repeat) + " cmd '" + str(cmd) + "'")
					p = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True,universal_newlines=True, preexec_fn=os.setsid)
					try:
						stdout, stderr = p.communicate(timeout=self.timeout_secs) # will raise exception if fail
						logging.debug("Benchmark output: " + str(stdout) + "... and stderr:" + str(stderr))
						logging.debug("Returncode: " + str(p.returncode))
						if stderr is None and p.returncode == 0:
							failure = False
						else:
							logging.error("Benchmark did not execute successfully for gpu stats collection.")
							os.killpg(os.getpgid(p.pid), signal.SIGKILL)
					except TimeoutExpired:
						logging.error("Timeout expired during execution of schedule for gpu stats collection.")
						os.killpg(os.getpgid(p.pid), signal.SIGKILL)
				except ProcessLookupError:
					logging.error("ProcessLookupError caught. Continuing.")

			line_results = []
			running_line = ""
			for char in stdout:
				if char == '\n':
					line_results.append(running_line)
					running_line = ""
				else:
					running_line = running_line + char

			for line in line_results:
				try:
					execution_time = float(line)
					break
				except ValueError:
					continue
			
			overall_execution_time += execution_time

		execution_time = overall_execution_time

		self.mean_gpu_execution_time = execution_time # TODO not properly collecting mean yet
		self.execution_time_variation = abs(self.mean_gpu_execution_time - self.mean_cpu_execution_time)

		with open(self.execution_times_filename, 'a') as f:
			f.write(schedule_identifier + ":" + str(execution_time) + "\n")

	def execute_benchmark(self, schedule_filename, allocated_tasks, episode_id, sample_idx=None, transition_idx=None, final_execution=True):

		cmd = "OPENSTREAM_SCHEDULE_FILE=" + str(schedule_filename) + " " + str(self.benchmark)

		schedule_identifier = ":".join([gpu_task.replace("\n","") for gpu_task in allocated_tasks])
		logging.debug("Executing trajectory " + str(schedule_identifier))

		final_identifier = ""
		if final_execution:
			final_identifier = " (final) "

		if sample_idx is not None:
			execution_identifier = str(episode_id) + " sample " + str(sample_idx) + " transition " + str(transition_idx) + final_identifier
		else:
			execution_identifier = str(episode_id) + final_identifier

		historical = False 
		# check if we can just use historical execution data
		if schedule_identifier in self.saved_execution_times and len(self.saved_execution_times[schedule_identifier]) >= self.target_total_execution_count:
			historical = True

			# I already have enough executions of this schedule, I can just use that data
			execution_time = np.random.choice(self.saved_execution_times[schedule_identifier], 1)[0] # select a random execution
			# add some gaussian noise based on the variation of times in the history
			# execution_time += (np.random.normal(0,np.std(self.saved_execution_times[schedule_identifier]),1)[0] * (0.25)) # half the noise because I don't have many samples...
			# NO NOISE, the execution time is fully deterministic
		
		else:

			overall_execution_time = 0.0
			for repeat in range(self.num_repeats):
				failure = True
				while failure:
					try:
						logging.debug("Running episode " + str(execution_identifier) + " cmd '" + str(cmd) + "'")
						p = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True,universal_newlines=True, preexec_fn=os.setsid)
						try:
							stdout, stderr = p.communicate(timeout=self.timeout_secs) # will raise exception if fail
							logging.debug("Benchmark output: " + str(stdout) + "... and stderr:" + str(stderr))
							logging.debug("Returncode: " + str(p.returncode))
							if stderr is None and p.returncode == 0:
								failure = False
							else:
								logging.error("Benchmark did not execute successfully (episode " + str(execution_identifier) + ").")
								os.killpg(os.getpgid(p.pid), signal.SIGKILL)
						except TimeoutExpired:
							logging.error("Timeout expired during execution of schedule for episode " + str(execution_identifier))
							os.killpg(os.getpgid(p.pid), signal.SIGKILL)
					except ProcessLookupError:
						logging.error("ProcessLookupError caught. Continuing.")

				line_results = []
				running_line = ""
				for char in stdout:
					if char == '\n':
						line_results.append(running_line)
						running_line = ""
					else:
						running_line = running_line + char

				for line in line_results:
					try:
						execution_time = float(line)
						break
					except ValueError:
						continue
				
				overall_execution_time += execution_time

			execution_time = overall_execution_time
			# this is always done sequentially so is fine
			with open(self.execution_times_filename, 'a') as f:
				f.write(schedule_identifier + ":" + str(execution_time) + "\n")

			if schedule_identifier in self.saved_execution_times:
				self.saved_execution_times[schedule_identifier].append(execution_time)
			else:
				self.saved_execution_times[schedule_identifier] = [execution_time]

		#reward = - ((execution_time - self.mean_center) / self.execution_time_variation)
		#reward = - execution_time
		reward = 1.0 - execution_time
		if historical:
			logging.info("Benchmark execution time at episode " + str(execution_identifier) + " (" + str(len(allocated_tasks)) + " tasks), taken from the historical executions, was " + str(execution_time) + ",reward was " + str(reward))
		else:
			logging.info("Benchmark execution time at episode " + str(execution_identifier) + " (" + str(len(allocated_tasks)) + " tasks) was " + str(execution_time) + ",reward was " + str(reward))

		return reward

	def take_action(self, action, episode_id, sample_idx=None, transition_idx=None, delay_execution=False):

		node_index = action.node_idx
		action_label = action.label

		self.current_state.feature_matrix[node_index][-2] = action_label
		self.current_state.feature_matrix[node_index][-1] = 1 # presence in partial solution
		self.current_state.partial_solution_node_indexes.append(node_index)

		if node_index in self.current_state.unselected_independent_node_indexes:
			self.current_state.unselected_independent_node_indexes = np.delete(self.current_state.unselected_independent_node_indexes, np.argwhere(self.current_state.unselected_independent_node_indexes == node_index))

		num_selected_nodes = len(self.current_state.partial_solution_node_indexes)
		
		requires_execution = False
		allocated_tasks = None

		if sample_idx is not None:
			episode_identifier = str(episode_id) + " sample " + str(sample_idx) + " transition " + str(transition_idx)
		else:
			episode_identifier = str(episode_id)

		# arbitrarily setting 20 so that we don't take an execution right before the end!
		if (num_selected_nodes == self.num_tasks or
					(
					self.partial_rewards != 0 and
					(num_selected_nodes-self.num_non_eligible_nodes) % int((self.num_tasks - self.num_non_eligible_nodes)/self.partial_rewards) == 0 and
					(self.num_tasks - self.num_non_eligible_nodes) >= ((num_selected_nodes-self.num_non_eligible_nodes) + int(0.1*(self.num_tasks - self.num_non_eligible_nodes)))
					)
				):

			if num_selected_nodes == self.num_tasks:
				logging.info("Episode " + str(episode_identifier) + " all tasks allocated.")
				schedule = self.current_state.get_schedule()
				termination = True
			else:
				logging.info("Episode " + str(episode_identifier) + " allocated " + str(num_selected_nodes-self.num_non_eligible_nodes) + " of " + str(self.num_tasks-self.num_non_eligible_nodes) + " tasks (" + "{0:.0f}".format(((num_selected_nodes-self.num_non_eligible_nodes)/(self.num_tasks-self.num_non_eligible_nodes))*100.0) + " %)")
				schedule = self.current_state.get_schedule(final=False)
				termination = False

			#if os.path.isfile(self.output_schedule_filename):
				# backup the previous episode's schedule
				# os.system("mv " + str(self.output_schedule_filename) + " " + str(self.output_schedule_filename) + str(episode_id-1))

			allocated_tasks = []

			if sample_idx is not None:
				schedule_filename = self.output_schedule_filename + "." + str(episode_id) + "." + str(sample_idx) + "." + str(transition_idx)
			else:
				schedule_filename = self.output_schedule_filename + "." + str(episode_id)

			with open(schedule_filename, 'w') as f:
				for task_idx in range(len(schedule)):
					if int(schedule[task_idx]) == 1:
						gpu_device_allocation = 0
						allocated_tasks.append(self.task_labels[task_idx])
						f.write(str(self.task_labels[task_idx]) + ":" + str(gpu_device_allocation) + "\n")

			if delay_execution:
				requires_execution = True
				reward = 0.0
			else:
				reward = self.execute_benchmark(schedule_filename, allocated_tasks, episode_id)

		else:

			if self.partial_rewards !=0 and (num_selected_nodes-self.num_non_eligible_nodes) % int((self.num_tasks - self.num_non_eligible_nodes)/self.partial_rewards) == 0:
				logging.info("Episode " + str(episode_identifier) + " allocated " + str(num_selected_nodes-self.num_non_eligible_nodes) + " of " + str(self.num_tasks-self.num_non_eligible_nodes) + " eligible tasks (" + "{0:.0f}".format(((num_selected_nodes-self.num_non_eligible_nodes)/(self.num_tasks-self.num_non_eligible_nodes))*100.0) + " %)")

			termination = False
			reward = 0.0 

		return self.current_state, reward, termination, requires_execution, allocated_tasks

	def reset(self):
		self.current_state = State(np.copy(self.base_state.feature_matrix),unselected_independent_node_indexes=np.copy(self.base_state.unselected_independent_node_indexes),partial_solution_node_indexes=np.copy(self.base_state.partial_solution_node_indexes).tolist())
